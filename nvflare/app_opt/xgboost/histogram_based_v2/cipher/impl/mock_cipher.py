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
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs import Decomposer

PUBLIC_KEY = "Public"
PRIVATE_KEY = "Private"
CONTEXT = "Context"


class MockEncryptedValue:
    """The combined int is too big for FOBS so it requires a special class"""
    def __init__(self, value):
        self.value = value


class MockEncryptedValueDecomposer(Decomposer):

    def supported_type(self):
        return MockEncryptedValue

    def decompose(self, target: MockEncryptedValue, datum_manager=None) -> Any:
        num_bytes = (target.value.bit_length() + 7) // 8 + 1
        return target.value.to_bytes(num_bytes, byteorder="big", signed=True)

    def recompose(self, data: Any, datum_manager=None) -> MockEncryptedValue:
        value = int.from_bytes(data, byteorder="big", signed=True)
        return MockEncryptedValue(value)


class MockHomomorphicCipher(HomomorphicCipher):

    def __init__(self):
        self.public_key = None
        self.private_key = None
        fobs.register(MockEncryptedValueDecomposer)

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

    def get_context_blob(self) -> bytes:
        # For this mock implementation, we use the same blob
        return (CONTEXT + str(self.public_key[1])).encode("utf-8")

    def set_context(self, context_blob: bytes):
        key_version = int(context_blob[len(CONTEXT):])
        self.public_key = PUBLIC_KEY, key_version
        self.private_key = PRIVATE_KEY, key_version

    def get_public_key_blob(self) -> bytes:
        return (PUBLIC_KEY + str(self.public_key[1])).encode("utf-8")

    def set_public_key(self, public_key_blob: bytes):
        public_key = PUBLIC_KEY, int(public_key_blob[len(PUBLIC_KEY):].decode("utf-8"))
        if public_key != self.public_key:
            self.public_key = public_key
            self.private_key = None

    def encrypt(self, value: float) -> Any:
        if not self.public_key:
            raise RuntimeError("Can't encrypt, no public key")

        return MockEncryptedValue(value)

    def decrypt(self, ciphertext: Any) -> float:
        if not self.private_key:
            raise RuntimeError("Can't decrypt, no private key")

        if ciphertext == 0:
            return 0

        if not isinstance(ciphertext, MockEncryptedValue):
            raise RuntimeError(f"Value of type {type(ciphertext)} is not encrypted: {ciphertext}")
        return ciphertext.value

    def add(self, a: Any, b: Any) -> Any:
        if not self.public_key:
            raise RuntimeError("Can't add, no public key")

        value = self._get_number(a) + self._get_number(b)
        if isinstance(a, MockEncryptedValue) or isinstance(b, MockEncryptedValue):
            return MockEncryptedValue(value)
        else:
            return value

    @staticmethod
    def _get_number(value: Any):
        if isinstance(value, MockEncryptedValue):
            return value.value
        else:
            return value
