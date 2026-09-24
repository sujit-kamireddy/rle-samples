# RLE BYOH data-code agent training demo (PowerShell)

Assumes the current RLE extension is installed. Also requires `az`, Git,
Python 3, access to the Foundry project below, and access to the Loom cookbook.

## 1. Sync `rle-samples` and set local variables

```powershell
azd ai rle version
az login
$demoRoot = Join-Path $HOME "rle-demo"
New-Item -ItemType Directory -Path $demoRoot -Force | Out-Null
git clone "https://github.com/sujit-kamireddy/rle-samples.git" "$demoRoot\rle-samples"
# If already cloned instead: git -C "$demoRoot\rle-samples" pull --ff-only origin main
Set-Location "$demoRoot\rle-samples\examples\harness\byoh\data_code_agent\rle"

$env:AZD_AI_RLE_ENABLE_ALL = "true"
$env:FOUNDRY_PROJECT_ENDPOINT = "https://ksujit-rle-eastus2-resource.services.ai.azure.com/api/projects/ksujit-rle-eastus2"
$env:LOOM_LOGS_ROOT = Join-Path $HOME "loom-runs"
```

## 2. Hosted BYOH endpoint

The sample's `rle.toml` already has:

```toml
[rle]
name = "data_code_agent_byoh"
version = "1.0.0"
type = "Harness"
subtype = "BYOH"
baseUrl = "https://data-code-agent-sujit-107203347434.us-east1.run.app/invoke"
```

This is the **hosted harness `/invoke` URL**, not the training API. Run
`azd ai rle show --output json` from `rle/` and confirm version `1.0.0` is
**Ready** in the selected project; it is already published, so do not
republish it. The separately discussed Foundry `code-repair-agent` Responses
endpoint was **not** used here and cannot simply replace the BYOH `/invoke`
URL. To register your own hosted harness, deploy the sample's `agent/`,
update `baseUrl`, choose a new RLE name/version, and follow the
[sample README's publish and ACR setup](./byoh/data_code_agent/README.md).
Only that path needs `AZURE_CONTAINER_REGISTRY_ENDPOINT`.

## 3. Training data and job

The fresh clone includes
`examples\harness\byoh\data_code_agent\job_data\train.jsonl` (1,000 tasks)
and `validation.jsonl` (200 disjoint tasks). Each JSONL row contains
`task` for the RLE grader/reset and `agent_input` for the harness:

```json
{"task":{"split":"FineEnvs/data-agent-harbor-train","task_index":0},"agent_input":{"split":"FineEnvs/data-agent-harbor-train","task_index":0}}
```

From `rle/`, `[train]` in `rle.toml` points to those files as
`../job_data/train.jsonl` and `../job_data/validation.jsonl`; the CLI uploads
them automatically. It also sets model `qwen3-32b-1`, `group_size = 4`, and
`max_concurrent_rollouts = 8`. Override the files for one job with
`--training-file` and `--validation-file` if needed.

```powershell
azd ai rle train --task-count 20 --follow
```

`--follow` mirrors files to `$HOME\loom-runs\rle-harness\<job-id>\` and
opens the separate rollout monitor. **Caveat:** earlier multi-task runs
hit the recipe's aggregate input-token limit; confirm the Vienna fix has
reached the service before relying on a 20-task demo.

## 4. Loom dashboard

In a second PowerShell terminal:

```powershell
$demoRoot = Join-Path $HOME "rle-demo"
git clone "https://github.com/microsoft-foundry/finetuning-cookbook.git" "$demoRoot\finetuning-cookbook"
Set-Location "$demoRoot\finetuning-cookbook"
python dashboard_server.py --root (Join-Path $HOME "loom-runs") --port 8000
```

Open **http://127.0.0.1:8000/** and select the `rle-harness/ftjob-...` run.
The cookbook clone requires repository access.
