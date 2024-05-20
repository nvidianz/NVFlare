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
import random
from typing import Any, Optional

from nvflare.app_opt.xgboost.histogram_based_v2.cipher.he_cipher import HomomorphicCipher

PUBLIC_KEY = "Public"
PRIVATE_KEY = "Private"


class MockHomomorphicCipher(HomomorphicCipher):

    def __init__(self):
        self.public_key = None
        self.private_key = None

    def name(self):
        return "mock"

    def initialize(self, parameters: Optional[dict] = None):
        pass

    def shutdown(self):
        pass

    def generate_keys(self, parameters: Optional[dict] = None):
        key_version = random.randint(0, 1000000)
        self.public_key = (PUBLIC_KEY, key_version)
        self.private_key = (PRIVATE_KEY, key_version)

    def get_public_key_blob(self) -> bytes:
        return str(self.public_key[1]).encode("utf-8")

    def set_public_key(self, public_key_blob: bytes):
        public_key = PUBLIC_KEY, int(public_key_blob.decode("utf-8"))
        if public_key != self.public_key:
            self.public_key = public_key
            self.private_key = None

    def encrypt(self, value: float) -> Any:
        if not self.public_key:
            raise RuntimeError("Can't encrypt, no public key")

        return self.public_key, value

    def decrypt(self, ciphertext: Any) -> float:
        if not self.private_key:
            raise RuntimeError("Can't encrypt, no private key")
        public_key, value = ciphertext
        if public_key != self.public_key:
            raise RuntimeError("Unmatched keys, can't decrypt")
        return value

    def add(self, a: Any, b: Any) -> Any:
        if not self.public_key:
            raise RuntimeError("Can't add, no public key")

        return self._get_number(a) + self._get_number(b)

    @staticmethod
    def _get_number(value: Any):
        if isinstance(value, tuple):
            return value[1]
        else:
            return value
