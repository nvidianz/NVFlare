# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Single-node DDP worker launched by the published Collab client method."""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-steps", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    import torch.distributed as dist
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision=args.revision,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        use_safetensors=True,
    ).to(local_rank)
    model.load_state_dict(load_file(args.input, device="cpu"))
    model.config.use_cache = False
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.Adafactor(model.parameters(), lr=1e-5)
    tokens = torch.randint(0, model.module.config.vocab_size, (1, 128), device=local_rank)
    for _ in range(args.local_steps):
        output = model(input_ids=tokens, labels=tokens)
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if rank == 0:
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in model.module.state_dict().items()}, args.output
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
