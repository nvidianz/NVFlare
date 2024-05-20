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

from nvflare.app_opt.xgboost.histogram_based_v2.cipher.cipher_loader import loader
from nvflare.app_opt.xgboost.histogram_based_v2.cipher.he_cipher import HomomorphicCipher


class Encryptor:
    def __init__(self, cipher: HomomorphicCipher, max_workers=10):
        self.max_workers = max_workers
        self.cipher = cipher
        self.exe = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    def encrypt(self, numbers):
        """
        Encrypt a list of clear text numbers

        Args:
            numbers: clear text numbers to be encrypted

        Returns: list of encrypted numbers

        """

        num_values = len(numbers)
        if num_values <= self.max_workers:
            w_values = [numbers]
            workers_needed = 1
        else:
            workers_needed = self.max_workers
            w_values = [None] * workers_needed
            n = int((num_values + workers_needed - 1) / workers_needed)
            w_load = [n] * workers_needed

            start = 0
            for i in range(workers_needed):
                end = start + w_load[i]
                w_values[i] = numbers[start:end]
                start = end

        total_count = 0
        for v in w_values:
            total_count += len(v)
        assert total_count == num_values

        items = []
        for i in range(workers_needed):
            items.append((self.cipher.name(), w_values[i]))
        return self._encrypt(items)

    def _encrypt(self, items):
        results = self.exe.map(_do_enc, items)
        rl = []
        for r in results:
            rl.extend(r)
        return rl


def _do_enc(item):
    cipher_name, numbers = item
    cipher = loader.load(cipher_name)
    return cipher.encrypt_vector(numbers)
