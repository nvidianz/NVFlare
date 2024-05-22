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

import concurrent.futures
import functools

from nvflare.app_opt.xgboost.histogram_based_v2.cipher.cipher_loader import loader
from nvflare.app_opt.xgboost.histogram_based_v2.cipher.he_cipher import HomomorphicCipher


class Decrypter:
    def __init__(self, cipher: HomomorphicCipher, max_workers=10):
        self.max_workers = max_workers
        self.cipher = cipher
        self.exe = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    def decrypt(self, encrypted_number_groups):
        """
        Encrypt a list of clear text numbers

        Args:
            encrypted_number_groups: list of lists of encrypted numbers to be decrypted

        Returns: list of lists of decrypted numbers

        """
        cipher_name = self.cipher.name()
        partial_func = functools.partial(_do_decrypt, cipher_name, self.cipher.get_context_blob())
        results = self.exe.map(partial_func, encrypted_number_groups)
        rl = []
        for r in results:
            rl.append(r)
        return rl


def _do_decrypt(cipher_name, context_blob, item):
    cipher = loader.find(cipher_name)
    cipher.set_context(context_blob)
    return cipher.decrypt_vector(item)
