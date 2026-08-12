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

"""Multi-GPU language-model benchmark implemented with the Collab API.

The server calls ``collab.clients.train`` directly.  Each client stages the
received safetensors state and launches a local multi-GPU ``torchrun`` worker.
The returned state dict deliberately exercises Collab's large-tensor call path.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch

from nvflare.collab import CollabRecipe, collab
from nvflare.recipe import ProdEnv

SMOKE_MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
GPT_J_MODEL_NAME = "EleutherAI/gpt-j-6b"
GPT_J_SAFETENSORS_REVISION = "f3f428825b6fc4c087af475ea729ac652edeee33"


def _state_dict(model):
    return {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}


def _load_initial_state(model_name, revision):
    from transformers import AutoModelForCausalLM

    model_args = {}
    if revision:
        model_args["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        use_safetensors=True,
        **model_args,
    )
    return _state_dict(model)


def _average(states):
    names = states[0].keys()
    averaged = {}
    for name in names:
        result = states[0][name].float()
        for state in states[1:]:
            result.add_(state[name].float())
        averaged[name] = (result / len(states)).to(states[0][name].dtype)
    return averaged


def _worker_path():
    """Locate the worker both in the source tree and Collab's custom folder."""
    source_path = Path(__file__).with_name("train.py")
    if source_path.is_file():
        return source_path
    return Path(__file__).parent / "dev_tools" / "collab" / "gpt_j_full_sft" / "train.py"


class GPTJClient:
    @collab.publish
    def train(self, global_state, round_number):
        """Run one local DDP training round and return a safetensors-only update."""
        from safetensors.torch import load_file, save_file

        site_name = collab.site_name
        nproc = int(collab.get_app_prop("nproc_per_node", 0)) or _gpu_count()
        if nproc < 2:
            raise RuntimeError(f"{site_name} needs at least two CUDA GPUs; found {nproc}")

        root = Path(collab.get_app_prop("work_root", "/tmp/nvflare/collab-gpt-j")) / site_name / f"round-{round_number}"
        root.mkdir(parents=True, exist_ok=True)
        input_path, output_path = root / "global.safetensors", root / "update.safetensors"
        save_file(global_state, input_path)
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={nproc}",
            str(_worker_path()),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--model-name",
            collab.get_app_prop("model_name", SMOKE_MODEL_NAME),
            "--local-steps",
            str(collab.get_app_prop("local_steps", 2)),
        ]
        revision = collab.get_app_prop("revision")
        if revision:
            command.extend(("--revision", revision))
        started = time.perf_counter()
        subprocess.run(command, check=True)
        elapsed = time.perf_counter() - started
        updated_state = load_file(output_path, device="cpu")
        return updated_state, {"site": site_name, "round": round_number, "torchrun_seconds": elapsed, "nproc": nproc}


class GPTJServer:
    @collab.main
    def run(self):
        rounds = collab.get_app_prop("num_rounds", 1)
        state = _load_initial_state(
            collab.get_app_prop("model_name", SMOKE_MODEL_NAME), collab.get_app_prop("revision")
        )
        for round_number in range(1, rounds + 1):
            started = time.perf_counter()
            results = collab.clients.train(state, round_number)
            failures = dict(results.failures)
            if failures:
                raise RuntimeError(f"Collab client failures in round {round_number}: {failures}")
            state = _average([update for update, _metrics in results.values()])
            print(
                f"NVFLARE_METRIC {{'event': 'round_complete', 'round': {round_number}, "
                f"'seconds': {time.perf_counter() - started:.3f}, 'clients': {len(results)}}}"
            )
        return state


def _gpu_count():
    import torch

    return torch.cuda.device_count()


def make_recipe(args):
    recipe = CollabRecipe(
        job_name="collab-gpt-j-6b-full-sft",
        server=GPTJServer(),
        client=GPTJClient(),
        min_clients=2,
        # Large model transfer is expected to take minutes, not the default minute.
        sync_task_timeout=args.sync_task_timeout,
    )
    recipe.set_server_prop("num_rounds", args.num_rounds)
    recipe.set_server_prop("model_name", args.model_name)
    recipe.set_server_prop("revision", args.revision)
    recipe.set_per_site_config(
        {
            args.client_ids[0]: {"nproc_per_node": args.site1_gpus, **_client_props(args)},
            args.client_ids[1]: {"nproc_per_node": args.site2_gpus, **_client_props(args)},
        }
    )
    # CollabRecipe ships the decorated module automatically.  The DDP worker
    # is a separate executable, so explicitly bundle it in every client app.
    recipe.add_client_file(str(Path(__file__).with_name("train.py")), clients=args.client_ids)
    return recipe


def _client_props(args):
    return {
        "local_steps": args.local_steps,
        "model_name": args.model_name,
        "revision": args.revision,
        "work_root": args.work_root,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Collab API multi-GPU language-model benchmark")
    parser.add_argument("--client-ids", nargs=2, default=["site-1", "site-2"])
    parser.add_argument("--site1-gpus", type=int, default=4)
    parser.add_argument("--site2-gpus", type=int, default=2)
    parser.add_argument("--model-name", default=SMOKE_MODEL_NAME)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--num-rounds", type=int, default=1)
    parser.add_argument("--local-steps", type=int, default=2)
    parser.add_argument("--sync-task-timeout", type=int, default=1800)
    parser.add_argument("--work-root", default="/tmp/nvflare/collab-gpt-j")
    parser.add_argument("--job-dir", default="/tmp/nvflare/collab-gpt-j/job")
    parser.add_argument("--startup-kit-location")
    parser.add_argument("--username", default="admin@nvidia.com")
    return parser.parse_args()


def main():
    args = parse_args()
    recipe = make_recipe(args)
    if args.startup_kit_location:
        run = recipe.execute(ProdEnv(startup_kit_location=args.startup_kit_location, username=args.username))
        print(f"Job status: {run.get_status()}")
    else:
        recipe.export(args.job_dir)
        print(f"Exported job to {args.job_dir}")


if __name__ == "__main__":
    main()
