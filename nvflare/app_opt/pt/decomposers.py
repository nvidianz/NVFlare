# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from io import BytesIO
from typing import Any

import torch
from safetensors.torch import save, load

from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.datum import DatumManager


class SerializationModule(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.register_buffer("saved_tensor", tensor)


class TensorDecomposer(fobs.Decomposer):
    def supported_type(self):
        return torch.Tensor

    def decompose(self, target: torch.Tensor, manager: DatumManager = None) -> Any:
        if target.dtype == torch.bfloat16:
            buffer = self._jit_serialize(target)
        else:
            buffer = self._safetensors_serialize(target)

        return {
            "buffer": buffer,
            "dtype": str(target.dtype),
        }

    def recompose(self, data: Any, manager: DatumManager = None) -> torch.Tensor:
        use_jit = False
        if isinstance(data, dict):
            if data["dtype"] == "torch.bfloat16":
                user_jit = True
            buf = data["buffer"]
        else:
            buf = data

        if use_jit:
            tensor = self._jit_deserialize(buf)
        else:
            tensor = self._safetensors_deserialize(buf)

        return tensor

    @staticmethod
    def _safetensors_serialize(tensor: torch.Tensor) -> bytes:
        state_dict = {"tensor": tensor}
        return save(state_dict)

    @staticmethod
    def _safetensors_deserialize(data: Any) -> torch.Tensor:
        state_dict = load(data)
        if not isinstance(state_dict, dict):
            raise ValueError(f"Invalid state_dict format: {type(state_dict)}")

        return state_dict.get("tensor", None)

    @staticmethod
    def _jit_serialize(tensor: torch.Tensor) -> bytes:
        stream = BytesIO()
        # unsupported ScalarType by numpy, use torch.jit to avoid Pickle
        module = SerializationModule(tensor)
        torch.jit.save(torch.jit.script(module), stream)
        return stream.getvalue()

    @staticmethod
    def _jit_deserialize(data: Any) -> torch.Tensor:
        stream = BytesIO(data)
        loaded_module = torch.jit.load(stream)
        return loaded_module.saved_tensor
