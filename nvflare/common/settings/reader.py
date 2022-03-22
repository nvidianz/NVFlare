# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional


"""Spec for settings reader
"""


class SettingsReader(ABC):

    readers = {}

    def __new__(cls, *args, **kwargs):
        file = args[0]
        _, ext = os.path.splitext(file)
        ext = ext.lstrip(".")
        if ext in SettingsReader.readers:
            reader_class = SettingsReader.readers[ext]
            return super().__new__(reader_class)
        else:
            raise NotImplementedError(f"No reader found for extension {ext}")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        SettingsReader.register_reader(cls)

    def __init__(self, file: str):
        self.file = file

    @staticmethod
    @abstractmethod
    def supported_extensions() -> List[str]:
        """File extensions supported by this reader

           Returns a list of extensions
           """
        pass

    @abstractmethod
    def read(self) -> Optional[Dict[str, Any]]:
        """Read a settings file

           Returns the file content as a dict
           """
        pass

    @staticmethod
    def register_reader(reader_class):
        for ext in reader_class.supported_extensions():
            SettingsReader.readers[ext] = reader_class
