# Collab API multi-GPU language-model benchmark

This benchmark uses the 135M-parameter `HuggingFaceTB/SmolLM2-135M` model by
default, so it can validate the Collab API and local multi-GPU DDP path on
16 GB A16 clients. One server calls two published client methods, and each
client runs local DDP through `torchrun`. The complete BF16 state dict is an
ordinary Collab call argument and return value, so the run measures the Collab
tensor-transfer path.

## Target topology

| Role | Host | GPUs |
| --- | --- | --- |
| Server | `a4u8g-mil-0020` | 4 x L20 (not used for training) |
| site-1 | `ipp1-1878` | 4 x A16 |
| site-2 | `ipp1-1895` | 4 x A16 |

Install NVFlare from this checkout and the requirements on the server and both
clients. SmolLM2 has a safetensors checkpoint; this benchmark never uses
`torch.load` or `torch.save`.

```bash
python -m pip install -e .
python -m pip install -r dev_tools/collab/gpt_j_full_sft/requirements.txt
python -m dev_tools.collab.gpt_j_full_sft.job \
  --client-ids site-1 site-2 --site1-gpus 4 --site2-gpus 2 \
  --job-dir /tmp/nvflare/collab-gpt-j/job
```

The default command exports a production job to
`/tmp/nvflare/collab-gpt-j/job`. Submit it once the server and the two clients
from the startup kit are running:

```bash
python -m dev_tools.collab.gpt_j_full_sft.job \
  --client-ids site-1 site-2 --site1-gpus 4 --site2-gpus 2 \
  --startup-kit-location /path/to/admin/startup-kit
```

The smoke-test defaults are one federated round and two optimizer steps. The
published call has a 30-minute timeout. `NVFLARE_METRIC` round records report
end-to-end Collab-call timing; `torchrun` timing is included in each client
response for comparison.

To reproduce the original full-model GPT-J workload on clients with at least
45 GB per GPU, pass its safetensors revision explicitly:

```bash
--model-name EleutherAI/gpt-j-6b \\
--revision f3f428825b6fc4c087af475ea729ac652edeee33
```
