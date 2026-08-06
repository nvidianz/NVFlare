# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgpack

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, MessageType
from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.sfm.constants import Types
from nvflare.fuel.f3.sfm.prefix import PREFIX_LEN, Prefix
from nvflare.fuel.f3.stream_cell import StreamCell
from nvflare.fuel.f3.streaming.byte_receiver import RxTask
from nvflare.fuel.f3.streaming.stream_const import STREAM_CHANNEL, STREAM_DATA_TOPIC, StreamDataType, StreamHeaderKey

CHUNK_SIZE = 1024
WINDOW_SIZE = 64 * CHUNK_SIZE
FRAME_COUNT = 22
DELAYED_FRAME_COUNT = 20
ORIGIN = "reordering_sender"
DESTINATION = "reordering_receiver"
APP_CHANNEL = "reordering_test"
APP_TOPIC = "blob"
STREAM_ID = 73093


def _make_stream_frame(seq: int) -> bytearray:
    headers = {
        MessageHeaderKey.MSG_TYPE: MessageType.REQ,
        MessageHeaderKey.REPLY_EXPECTED: False,
        MessageHeaderKey.CHANNEL: STREAM_CHANNEL,
        MessageHeaderKey.TOPIC: STREAM_DATA_TOPIC,
        MessageHeaderKey.ORIGIN: ORIGIN,
        MessageHeaderKey.DESTINATION: DESTINATION,
        MessageHeaderKey.FROM_CELL: ORIGIN,
        MessageHeaderKey.TO_CELL: DESTINATION,
        MessageHeaderKey.OPTIONAL: False,
        MessageHeaderKey.SECURE: False,
        StreamHeaderKey.CHANNEL: APP_CHANNEL,
        StreamHeaderKey.TOPIC: APP_TOPIC,
        StreamHeaderKey.SIZE: FRAME_COUNT * CHUNK_SIZE,
        StreamHeaderKey.STREAM_ID: STREAM_ID,
        StreamHeaderKey.DATA_TYPE: StreamDataType.FINAL if seq == FRAME_COUNT - 1 else StreamDataType.CHUNK,
        StreamHeaderKey.SEQUENCE: seq,
        StreamHeaderKey.OFFSET: seq * CHUNK_SIZE,
        StreamHeaderKey.RELIABLE: False,
        StreamHeaderKey.OPTIONAL: False,
        StreamHeaderKey.CHUNK_SIZE: CHUNK_SIZE,
        StreamHeaderKey.WINDOW_SIZE: WINDOW_SIZE,
    }
    if seq == 0:
        headers[StreamHeaderKey.ACK_INTERVAL] = WINDOW_SIZE

    encoded_headers = msgpack.packb(headers)
    payload = bytes([seq]) * CHUNK_SIZE
    prefix = Prefix(
        length=PREFIX_LEN + len(encoded_headers) + len(payload),
        header_len=len(encoded_headers),
        type=Types.DATA,
        app_id=CoreCell.APP_ID,
    )
    frame = bytearray(prefix.length)
    prefix.to_buffer(frame, 0)
    frame[PREFIX_LEN : PREFIX_LEN + len(encoded_headers)] = encoded_headers
    frame[PREFIX_LEN + len(encoded_headers) :] = payload
    return frame


def test_conn_manager_handles_sequence_zero_delayed_behind_more_than_16_frames(monkeypatch):
    monkeypatch.setattr(CommConfigurator, "get_streaming_chunk_size", lambda self, default: CHUNK_SIZE)
    monkeypatch.setattr(CommConfigurator, "get_streaming_window_size", lambda self, default: WINDOW_SIZE)
    monkeypatch.setattr(CommConfigurator, "get_streaming_ack_interval", lambda self, default: WINDOW_SIZE)
    monkeypatch.setattr(CommConfigurator, "get_streaming_max_out_seq_chunks", lambda self, default: default)

    cell = CoreCell(DESTINATION, "tcp://localhost:1", secure=False, credentials={})
    stream_cell = StreamCell(cell)
    cell.fire_and_forget = MagicMock(return_value={})

    seq_zero_started = threading.Event()
    release_seq_zero = threading.Event()
    delayed_frames_processed = threading.Event()
    receive_done = threading.Event()
    received = bytearray()
    processed_count = 0
    processed_lock = threading.Lock()
    original_process_message = cell.process_message

    def delayed_process_message(endpoint, connection, app_id, message):
        nonlocal processed_count
        seq = message.get_header(StreamHeaderKey.SEQUENCE)
        if message.get_header(MessageHeaderKey.TOPIC) == STREAM_DATA_TOPIC and seq == 0:
            seq_zero_started.set()
            release_seq_zero.wait(timeout=5)

        original_process_message(endpoint, connection, app_id, message)

        if message.get_header(MessageHeaderKey.TOPIC) == STREAM_DATA_TOPIC and seq > 0:
            with processed_lock:
                processed_count += 1
                if processed_count >= DELAYED_FRAME_COUNT:
                    delayed_frames_processed.set()

    def receive_stream(_future, stream, _resume):
        try:
            while True:
                data = stream.read(4 * CHUNK_SIZE)
                if not data:
                    break
                received.extend(data)
        finally:
            stream.close()
            receive_done.set()

    stream_cell.register_stream_cb(APP_CHANNEL, APP_TOPIC, receive_stream)
    cell.process_message = delayed_process_message

    connection = MagicMock()
    connection.get_conn_properties.return_value = {}
    sfm_conn = SimpleNamespace(
        conn=connection,
        sfm_endpoint=SimpleNamespace(endpoint=Endpoint(ORIGIN)),
        get_name=lambda: "reordering_test_connection",
    )
    manager = cell.communicator.conn_manager

    try:
        manager.process_frame(sfm_conn, _make_stream_frame(0))
        assert seq_zero_started.wait(timeout=5)

        for seq in range(1, FRAME_COUNT):
            manager.process_frame(sfm_conn, _make_stream_frame(seq))

        assert delayed_frames_processed.wait(timeout=5)
        with RxTask.map_lock:
            task = RxTask.rx_task_map[(ORIGIN, STREAM_ID)]
            assert task.max_out_seq == 65
            assert len(task.out_seq_chunks) >= DELAYED_FRAME_COUNT
            assert task.error is None

        release_seq_zero.set()
        assert receive_done.wait(timeout=5)
        assert received == b"".join(bytes([seq]) * CHUNK_SIZE for seq in range(FRAME_COUNT))
    finally:
        release_seq_zero.set()
        cell.communicator.stop()
        CoreCell.ALL_CELLS.pop(DESTINATION, None)
        with RxTask.map_lock:
            task = RxTask.rx_task_map.pop((ORIGIN, STREAM_ID), None)
        if task and task.cleanup_timer:
            task.cleanup_timer.cancel()
