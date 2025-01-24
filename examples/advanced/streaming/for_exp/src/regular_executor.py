# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
import torch

class RegularExecutor(Executor):
    def __init__(self):
        Executor.__init__(self)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_info(fl_ctx, f"got task {task_name}")
        if task_name == "retrieve_dict":
            params = shareable.get("weight")
            self.log_info(fl_ctx, f"received container type: {type(params)} size: {len(params)}")
            torch.save(params, "model.pt")
            return make_reply(ReturnCode.OK)
        else:
            self.log_error(fl_ctx, f"got unknown task {task_name}")
            return make_reply(ReturnCode.TASK_UNKNOWN)
