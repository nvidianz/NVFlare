# Collab API GPT-J 6B benchmark

This benchmark recreates the three-host GPT-J full-model run with the Collab
API: one server calls two published client methods, and each client runs local
DDP through `torchrun`. The complete BF16 state dict is an ordinary Collab call
argument and return value, so the run measures the Collab tensor-transfer path.

## Target topology

| Role | Host | GPUs |
| --- | --- | --- |
| Server | `2u1g-x570-0286` | none |
| site-1 | `a4u8g-mil-0026` | 4 x RTX 5880 Ada |
| site-2 | `smc220-0008` | 2 x A40 |

Install NVFlare from this checkout and the requirements on the server and both
clients. The pinned GPT-J revision is safetensors-only; this benchmark never
uses `torch.load` or `torch.save`.

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
published call has a 30-minute timeout because it transfers a full 11.27 GiB
BF16 GPT-J state in each direction. `NVFLARE_METRIC` round records report
end-to-end Collab-call timing; `torchrun` timing is included in each client
response for comparison.
