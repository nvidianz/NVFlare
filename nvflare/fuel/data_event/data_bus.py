# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List

from nvflare.fuel.data_event.pub_sub import EventPubSub


class DataBus(EventPubSub):
    """
    Singleton class for a simple data bus implementation.

    This class allows components to subscribe to topics, publish messages to topics,
    and store/retrieve messages associated with specific keys and topics.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> "DataBus":
        """
        Create a new instance of the DataBus class.
        This method ensures that only one instance of the class is created (singleton pattern).
        The databus


        """
        with cls._lock:
            if not cls._instance:
                cls._instance = super(DataBus, cls).__new__(cls)
                cls._instance.subscribers = {}
                cls._instance.data_store = {}
        return cls._instance

    def subscribe(
        self,
        topics: List[str],
        callback: Callable[[str, Any, "DataBus"], None],
        **cb_kwargs,
    ) -> None:
        """
        Subscribe a callback function to one or more topics.

        Args:
            topics (List[str]): A list of topics to subscribe to.
            callback (Callable): The callback function to be called when messages are published to the subscribed topics.
        """

        if not topics:
            raise ValueError("topics must non-empty")

        for topic in topics:
            if topic.isspace():
                raise ValueError(f"topics {topics}contains white space topic")

            with self._lock:
                if topic not in self.subscribers:
                    self.subscribers[topic] = []
                self.subscribers[topic].append((callback, cb_kwargs))

    def unsubscribe(
        self,
        topic: str,
        callback=None,
    ) -> None:
        """Unsubscribe from the specified topic.
        If the callback is specified, only remove the subscription that has this callback;
        If the callback is not specified, remove all subscriptions of this topic.

        Args:
            topic: the topic to unsubscribe
            callback: the callback to be removed

        Returns: None

        """
        with self._lock:
            if topic not in self.subscribers:
                return

            if callback is None:
                # remove this topic
                self.subscribers.pop(topic, None)
                return

            subs_to_delete = []
            subs = self.subscribers[topic]
            assert isinstance(subs, list)
            for sub in subs:
                # sub is a tuple of (cb, cb_args)
                if sub[0] == callback:
                    subs_to_delete.append(sub)

            for sub in subs_to_delete:
                subs.remove(sub)

            if len(subs) == 0:
                # no more subs for this topic!
                self.subscribers.pop(topic, None)

    def publish(self, topics: List[str], datum: Any) -> None:
        """
        Publish a data to one or more topics, notifying all subscribed callbacks.

        Args:
            topics (List[str]): A list of topics to publish the data to.
            datum (Any): The data to be published to the specified topics.
        """
        if not topics:
            return

        # minimize the time of lock - only manage the subscribers data structure within the lock
        # do not run the CBs within the lock
        with self._lock:
            subs_to_execute = []
            for topic in topics:
                subscribers = self.subscribers.get(topic)
                if subscribers:
                    for sub in subscribers:
                        callback, kwargs = sub
                        subs_to_execute.append((topic, callback, kwargs))

        if not subs_to_execute:
            return

        executor = ThreadPoolExecutor(max_workers=len(subs_to_execute))
        for sub in subs_to_execute:
            topic, callback, kwargs = sub
            executor.submit(callback, topic, datum, self, **kwargs)
        executor.shutdown()

    def put_data(self, key: Any, datum: Any) -> None:
        """
        Store a data associated with a key and topic.

        Args:
            key (Any): The key to associate with the stored message.
            datum (Any): The message to be stored.
        """
        with self._lock:
            self.data_store[key] = datum

    def get_data(self, key: Any) -> Any:
        """
        Retrieve a stored data associated with a key and topic.

        Args:
            key (Any): The key associated with the stored message.

        Returns:
            Any: The stored datum if found, or None if not found.
        """
        return self.data_store.get(key)
