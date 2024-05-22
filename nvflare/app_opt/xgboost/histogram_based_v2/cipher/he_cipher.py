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
from abc import ABC, abstractmethod
from typing import Any, List, Optional


class HomomorphicCipher(ABC):
    """A homomorphic encryption cipher. Non-default constructors are not allowed since the instance is always created
    with default constructor by the CipherLoader. Use initialize() to provide parameters to the cipher.
    """

    @abstractmethod
    def name(self):
        """Return name of the cipher"""
        pass

    @abstractmethod
    def initialize(self, parameters: Optional[dict] = None):
        """Initialize the cipher and provide parameters

        Args:
            parameters: Optional parameters required by the cipher
        """
        pass

    @abstractmethod
    def shutdown(self):
        """Shutdown the cipher and free the resources"""
        pass

    @abstractmethod
    def generate_keys(self, parameters: Optional[dict] = None):
        """Generate keys for encryption/decryption

        Args:
            parameters: Optional parameters required by key generation
        """
        pass

    @abstractmethod
    def get_context_blob(self) -> bytes:
        """Get entire context (including private key)

        Returns:
            Serialized context
        """
        pass

    @abstractmethod
    def set_context(self, context_blob: bytes):
        """Set the context

        Args:
            context_blob: Serialized context
        """
        pass

    @abstractmethod
    def get_public_key_blob(self) -> bytes:
        """Get serialized public key as bytes

        Returns:
            Serialized public key
        """
        pass

    @abstractmethod
    def set_public_key(self, public_key_blob: bytes):
        """Set the public key for the cipher. The cipher with public key only cannot perform decryption

        Args:
            public_key_blob: Serialized public key
        """
        pass

    @abstractmethod
    def encrypt(self, value: float) -> Any:
        """Encrypt a value

        Args:
            value: A number in float or int
        Returns:
            Encrypted value in a cipher specific format
        """
        pass

    def encrypt_vector(self, values: List[float]) -> List[Any]:
        """Encrypt a vector. This method should be overridden if an
        implementation has more efficient way to encrypt a vector

        Args:
            values: A list of numbers in float or int
        Returns:
            A list of encrypted value in a cipher specific format
        """
        return [self.encrypt(x) for x in values]

    @abstractmethod
    def decrypt(self, ciphertext: Any) -> float:
        """Decrypt a ciphertext into float

        Args:
            ciphertext: A ciphertext to be decrypted
        Returns:
            Clear-text value as float or int
        """
        pass

    def decrypt_vector(self, ciphertexts: List[float]) -> List[float]:
        """Decrypt a vector of ciphertext. This method should be overridden if an
        implementation has more efficient way to decrypt a vector

        Args:
            ciphertexts: A list of ciphertext
        Returns:
            A list of float or int
        """
        return [self.decrypt(x) for x in ciphertexts]

    @abstractmethod
    def add(self, a: Any, b: Any) -> Any:
        """Add two values, either A or B can be in clear or cipher text

        Args:
            a: First operand in clear or cipher text
            b: Second operand in clear or cipher text
        Returns:
            The sum in ciphertext in cipher specific format
        """
        pass

    def sum(self, values: List[Any]) -> Any:
        """Add a list of numbers in ciphertext. This method should be overridden if an
        implementation has more efficient way to do the summation

        Args:
            values: A list of numbers in ciphertext
        Returns:
            The sum in ciphertext in cipher specific format
        """
        if len(values) < 2:
            raise ValueError("sum requires at least 2 entries")

        result = values[0]
        for value in values[1:]:
            result = self.add(result, value)

        return result
