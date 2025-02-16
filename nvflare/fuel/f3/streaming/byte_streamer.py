# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import threading
import time
from typing import Callable, Optional

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.stats_pool import StatsPoolManager
from nvflare.fuel.f3.streaming.stream_const import (
    STREAM_ACK_TOPIC,
    STREAM_CHANNEL,
    STREAM_DATA_TOPIC,
    StreamDataType,
    StreamHeaderKey,
)
from nvflare.fuel.f3.streaming.stream_types import Stream, StreamError, StreamFuture, StreamTaskSpec
from nvflare.fuel.f3.streaming.stream_utils import (
    ONE_MB,
    gen_stream_id,
    stream_stats_category,
    stream_thread_pool,
    wrap_view,
)

# Default settings
STREAM_CHUNK_SIZE = 1024 * 1024
STREAM_WINDOW_SIZE = 16 * STREAM_CHUNK_SIZE
STREAM_ACK_WAIT = 300
STREAM_RETRY_WAIT = 5
STREAM_RETRY_TIMEOUT = 60

STREAM_TYPE_BYTE = "byte"
STREAM_TYPE_BLOB = "blob"
STREAM_TYPE_FILE = "file"

COUNTER_NAME_SENT = "sent"

log = logging.getLogger(__name__)


class TxTask(StreamTaskSpec):
    def __init__(
        self,
        cell: CoreCell,
        chunk_size: int,
        channel: str,
        topic: str,
        target: str,
        headers: dict,
        stream: Stream,
        reliable: bool,
        secure: bool,
        optional: bool,
    ):
        self.cell = cell
        self.chunk_size = chunk_size
        self.sid = gen_stream_id()
        self.buffer = wrap_view(bytearray(chunk_size))
        # Optimization to send the original buffer without copying
        self.direct_buf: Optional[bytes] = None
        self.buffer_size = 0
        self.channel = channel
        self.topic = topic
        self.target = target
        self.headers = headers
        self.stream = stream
        self.stream_future = None
        self.task_future = None
        self.ack_waiter = threading.Event()
        self.seq = 0
        self.offset = 0
        self.offset_ack = 0
        self.reliable = reliable
        self.secure = secure
        self.optional = optional
        self.stopped = False
        self.stopping = False

        self.stream_future = StreamFuture(self.sid, task_handle=self)
        self.stream_future.set_size(stream.get_size())

        comm_config = CommConfigurator()
        self.window_size = comm_config.get_streaming_window_size(STREAM_WINDOW_SIZE)
        self.ack_wait = comm_config.get_streaming_ack_wait(STREAM_ACK_WAIT)
        self.retry_wait = comm_config.get_streaming_retry_wait(STREAM_RETRY_WAIT)
        self.retry_timeout = comm_config.get_streaming_retry_timeout(STREAM_RETRY_TIMEOUT)

        if self.reliable:
            self.pending_messages = {}
            self.retry_lock = threading.RLock()  # Need reentrant lock
            self.retry_task_future = stream_thread_pool.submit(self.retry_task)
        else:
            self.pending_messages = None
            self.retry_lock = None
            self.retry_task_future = None

    def __str__(self):
        return f"Tx[SID:{self.sid} to {self.target} for {self.channel}/{self.topic}]"

    def send_loop(self):
        """Read/send loop to transmit the whole stream with flow control"""

        while not self.stopped:
            buf = self.stream.read(self.chunk_size)
            if not buf:
                # End of Stream
                self.send_pending_buffer(final=True)
                self.stop()
                return

            # Flow control
            window = self.offset - self.offset_ack
            # It may take several ACKs to clear up the window
            while window >= self.window_size and not self.stopped:
                log.debug(f"{self} window size {window} exceeds limit: {self.window_size}")
                self.ack_waiter.clear()

                if not self.ack_waiter.wait(timeout=self.ack_wait):
                    self.stop(StreamError(f"{self} ACK timeouts after {self.ack_wait} seconds"))
                    return

                window = self.offset - self.offset_ack

            size = len(buf)
            if size > self.chunk_size:
                raise StreamError(f"{self} Stream returns invalid size: {size}")

            # Don't push out chunk when it's equal, wait till next round to detect EOS
            # For example, if the stream size is chunk size (1M), this avoids sending two chunks.
            if size + self.buffer_size > self.chunk_size:
                self.send_pending_buffer()

            if size == self.chunk_size:
                self.direct_buf = buf
            else:
                self.buffer[self.buffer_size : self.buffer_size + size] = buf
            self.buffer_size += size

    def send_pending_buffer(self, final=False):

        if self.buffer_size == 0:
            payload = bytes(0)
        elif self.buffer_size == self.chunk_size:
            if self.direct_buf:
                payload = self.direct_buf
            else:
                payload = self.buffer
        else:
            payload = self.buffer[0 : self.buffer_size]

        message = Message(None, payload)

        if self.headers:
            message.add_headers(self.headers)

        message.add_headers(
            {
                StreamHeaderKey.CHANNEL: self.channel,
                StreamHeaderKey.TOPIC: self.topic,
                StreamHeaderKey.SIZE: self.stream.get_size(),
                StreamHeaderKey.STREAM_ID: self.sid,
                StreamHeaderKey.DATA_TYPE: StreamDataType.FINAL if final else StreamDataType.CHUNK,
                StreamHeaderKey.SEQUENCE: self.seq,
                StreamHeaderKey.OFFSET: self.offset,
                StreamHeaderKey.RELIABLE: self.reliable,
                StreamHeaderKey.OPTIONAL: self.optional,
            }
        )

        if self.reliable:
            curr_time = time.time()
            # The tuple is start, last_retry, message
            with self.retry_lock:
                self.pending_messages[self.seq] = curr_time, curr_time, message
                # sanity check
                pending_size = sum(
                    len(msg.payload) if msg.payload else 0 for _, _, msg in self.pending_messages.values()
                )
                # Pending messages may exceed windows size by a few chunks due to the timeing of cleanup
                if pending_size > 2 * self.window_size:
                    log.error(f"Too many retry messages ({pending_size} > {self.window_size})")

        errors = self.cell.fire_and_forget(
            STREAM_CHANNEL, STREAM_DATA_TOPIC, self.target, message, secure=self.secure, optional=self.optional
        )
        error = errors.get(self.target)
        if error:
            msg = f"{self} Message sending error to target {self.target}: {error}"
            if self.reliable:
                log.error(f"{msg}, will retry in {self.retry_wait} seconds")
            else:
                self.stop(StreamError(msg))
                return

        # Update state
        self.seq += 1
        self.offset += self.buffer_size
        self.buffer_size = 0
        self.direct_buf = None

        # Update future
        self.stream_future.set_progress(self.offset)

    def stop(self, error: Optional[StreamError] = None, notify=True):

        if self.stopped:
            return

        if not error and self.pending_messages:
            # Can't end stream if pending_messages are not empty
            self.stopping = True
            return

        self.stopped = True
        self.remove_task()
        if not self.ack_waiter.is_set():
            self.ack_waiter.set()

        if self.task_future:
            self.task_future.cancel()

        if not error:
            # Result is the number of bytes streamed
            if self.stream_future:
                self.stream_future.set_result(self.offset)
            return

        if self.reliable:
            with self.retry_lock:
                if self.pending_messages:
                    self.pending_messages.clear()

        # Error handling
        log.debug(f"{self} Stream error: {error}")
        if self.stream_future:
            self.stream_future.set_exception(error)

        if notify:
            message = Message(None, None)

            if self.headers:
                message.add_headers(self.headers)

            message.add_headers(
                {
                    StreamHeaderKey.STREAM_ID: self.sid,
                    StreamHeaderKey.DATA_TYPE: StreamDataType.ERROR,
                    StreamHeaderKey.OFFSET: self.offset,
                    StreamHeaderKey.ERROR_MSG: str(error),
                }
            )
            self.cell.fire_and_forget(
                STREAM_CHANNEL, STREAM_DATA_TOPIC, self.target, message, secure=self.secure, optional=True
            )

    def handle_ack(self, message: Message):

        origin = message.get_header(MessageHeaderKey.ORIGIN)
        ack_seq = message.get_header(StreamHeaderKey.SEQUENCE, None)
        offset = message.get_header(StreamHeaderKey.OFFSET, None)
        error = message.get_header(StreamHeaderKey.ERROR_MSG, None)

        if error:
            self.stop(StreamError(f"{self} Received error from {origin}: {error}"), notify=False)
            return

        if self.reliable and ack_seq is None:
            self.stop(StreamError(f"{self} Receiving end at {origin} doesn't support reliable streaming"), notify=True)
            return

        if offset > self.offset_ack:
            self.offset_ack = offset

        if self.reliable:
            with self.retry_lock:
                if self.pending_messages and ack_seq is not None:
                    for seq, value in list(self.pending_messages.items()):
                        if seq <= ack_seq:
                            del self.pending_messages[seq]

            if self.stopping and not self.pending_messages:
                self.stop()

        if not self.ack_waiter.is_set():
            self.ack_waiter.set()

    def start_task_thread(self, task_handler: Callable):
        self.task_future = stream_thread_pool.submit(task_handler, self)

    def cancel(self):
        self.stop(error=StreamError("cancelled"))

    def retry_task(self):
        try:
            while not self.stopped:
                with self.retry_lock:
                    if not self.pending_messages:
                        if self.stopping:
                            self.stop()
                        continue

                    curr_time = time.time()
                    for seq, value in self.pending_messages.items():
                        start_time, last_retry, message = value
                        retry_time = curr_time - start_time
                        if retry_time > self.retry_timeout:
                            msg = f"{self} Seq {seq} retry failed after trying for {retry_time} seconds"
                            log.error(msg)
                            self.stop(error=StreamError(msg))
                            break

                        wait_time = curr_time - last_retry
                        if wait_time < self.retry_wait:
                            continue

                        errors = self.cell.fire_and_forget(
                            STREAM_CHANNEL,
                            STREAM_DATA_TOPIC,
                            self.target,
                            message,
                            secure=self.secure,
                            optional=self.optional,
                        )
                        error = errors.get(self.target)
                        if error:
                            log.error(
                                f"{self} Message sending error to target "
                                f"{self.target}: {error}, will retry again in {self.retry_wait} seconds"
                            )

                        self.pending_messages[seq] = start_time, curr_time, message

                time.sleep(self.retry_wait)

        except Exception as ex:
            log.error(f"{self} retry thread ended due to error: {ex}")

    def remove_task(self):
        with ByteStreamer.map_lock:
            ByteStreamer.tx_task_map.pop(self.sid, None)
            log.debug(f"{self} is removed")


class ByteStreamer:

    tx_task_map = {}
    map_lock = threading.Lock()

    sent_stream_counter_pool = StatsPoolManager.add_counter_pool(
        name="Sent_Stream_Counters",
        description="Counters of sent streams",
        counter_names=[COUNTER_NAME_SENT],
    )

    sent_stream_size_pool = StatsPoolManager.add_msg_size_pool("Sent_Stream_Sizes", "Sizes of streams sent (MBs)")

    def __init__(self, cell: CoreCell):
        self.cell = cell
        self.cell.register_request_cb(channel=STREAM_CHANNEL, topic=STREAM_ACK_TOPIC, cb=self._ack_handler)
        self.chunk_size = CommConfigurator().get_streaming_chunk_size(STREAM_CHUNK_SIZE)

    def get_chunk_size(self):
        return self.chunk_size

    def send(
        self,
        channel: str,
        topic: str,
        target: str,
        headers: dict,
        stream: Stream,
        stream_type=STREAM_TYPE_BYTE,
        reliable=True,
        secure=False,
        optional=False,
    ) -> StreamFuture:
        tx_task = TxTask(
            self.cell, self.chunk_size, channel, topic, target, headers, stream, reliable, secure, optional
        )
        with ByteStreamer.map_lock:
            ByteStreamer.tx_task_map[tx_task.sid] = tx_task

        tx_task.start_task_thread(self._transmit_task)

        fqcn = self.cell.my_info.fqcn
        ByteStreamer.sent_stream_counter_pool.increment(
            category=stream_stats_category(fqcn, channel, topic, stream_type), counter_name=COUNTER_NAME_SENT
        )

        ByteStreamer.sent_stream_size_pool.record_value(
            category=stream_stats_category(fqcn, channel, topic, stream_type), value=stream.get_size() / ONE_MB
        )

        return tx_task.stream_future

    @staticmethod
    def _transmit_task(task: TxTask):

        try:
            task.send_loop()
        except Exception as ex:
            msg = f"{task} Error while sending: {ex}"
            log.error(msg)
            task.stop(StreamError(msg), True)

    @staticmethod
    def _ack_handler(message: Message):

        sid = message.get_header(StreamHeaderKey.STREAM_ID)
        with ByteStreamer.map_lock:
            tx_task = ByteStreamer.tx_task_map.get(sid, None)

        if not tx_task:
            origin = message.get_header(MessageHeaderKey.ORIGIN)
            offset = message.get_header(StreamHeaderKey.OFFSET, None)
            seq = message.get_header(StreamHeaderKey.SEQUENCE, None)
            # Last few ACKs always arrive late for non-reliable streaming
            log.debug(f"ACK for stream {sid} received late from {origin} with offset {offset} seq {seq}")
            return

        tx_task.handle_ack(message)
