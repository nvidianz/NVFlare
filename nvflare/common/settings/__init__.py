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

import importlib
import os
from os.path import dirname, join

from .reader import SettingsReader
from .settings import Settings, RuntimeSettings, FlareSettings

# Load all readers by importing them
for module in os.listdir(join(dirname(__file__), "readers")):
    if module != '__init__.py' and module[-3:] == '.py':
        reader = ".readers." + module[:-3]
        importlib.import_module(reader, __package__)
