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
import importlib
import inspect
import logging
import os
from typing import Optional, Union, Type
from nvflare.app_opt.xgboost.histogram_based_v2.cipher import impl
from nvflare.app_opt.xgboost.histogram_based_v2.cipher.he_cipher import HomomorphicCipher

log = logging.getLogger(__name__)


class CipherLoader:
    """A loader for homomorphic ciphers"""

    def __init__(self):
        self._ciphers = {}
        self._class_cache = set()

        # Load all built-in ciphers in impl folder
        self.load(os.path.dirname(impl.__file__), impl.__package__)

    def find(self, name: str) -> Optional[HomomorphicCipher]:
        """Find cipher by name

        Args:
            name: Cipher name
        Returns:
            The cipher with the name. None if not found
        """
        return self._ciphers.get(name, None)

    def register(self, cipher: Union[HomomorphicCipher, Type[HomomorphicCipher]]) -> None:

        if inspect.isclass(cipher):
            cipher_instance = cipher()
        else:
            cipher_instance = cipher

        name = cipher_instance.name()
        if not name:
            raise ValueError(f"Cipher {type(cipher_instance)} has no name")

        self._ciphers[name] = cipher_instance

    def load(self, search_path: str, package_prefix: Optional[str] = None):
        """Load all the ciphers in the search path

        Args:
            search_path: The path to search
            package_prefix: The prefix of package relative to the search_path
        """
        for root, dirs, files in os.walk(search_path):
            for filename in files:
                if filename.endswith(".py") and not filename.startswith("__init__"):
                    module = filename[:-3]
                    sub_folder = root[len(search_path):]
                    if sub_folder:
                        sub_folder = sub_folder.strip("/").replace("/", ".")

                    if sub_folder:
                        module = sub_folder + "." + module

                    if package_prefix:
                        module = package_prefix + "." + module

                    imported = importlib.import_module(module)
                    for _, cls_obj in inspect.getmembers(imported, inspect.isclass):
                        if cls_obj.__name__ in self._class_cache:
                            continue
                        self._class_cache.add(cls_obj.__name__)

                        if issubclass(cls_obj, HomomorphicCipher) and not inspect.isabstract(cls_obj):
                            spec = inspect.getfullargspec(cls_obj.__init__)
                            if len(spec.args) == 1:
                                self.register(cls_obj)
                            else:
                                # Can't handle argument in constructor
                                log.warning(f"Invalid cipher, __init__ with extra arguments: {module}")


loader = CipherLoader()

if __name__ == "__main__":

    instance = loader.find("mock")
    print(f"Name: {instance.name()} Type: {type(instance)}")
    print("end")
