# RLE samples

Each sample is a self-hosted OpenEnv environment with a host-agnostic
`rle.toml` manifest. The manifest declares the immutable RLE identity and
callable interface:

```toml
[rle]
name = "code_rl"
version = "1.0.0"
type = "Gym"
subtype = "OpenEnv"
```

The `type` and `subtype` values match the RLE control plane. Current samples
use `Gym` and `OpenEnv`. The manifest intentionally does not contain Foundry
project endpoints, registry locations, service IDs, or credentials.

When a sample is initialized through `azd ai rle init <folder-name>`, the CLI
updates `rle.name` to the target folder name. Before publishing another
version, update `rle.version` to the direct next major, minor, or patch
release.

## `code_rl` compact dataset maintenance

`envs/code_rl` ships its 900-training-task and 100-validation-task snapshot
under `data/`, so normal Docker builds and runtime instances do not download
Hugging Face data. The snapshot generator is intentionally outside the sample
at `tools/generate_code_rl_compact_dataset.py`; it is not copied by
`azd ai rle init`.

Maintainers can intentionally refresh the revision-pinned snapshot and its
checksums from the repository root:

```bash
python tools/generate_code_rl_compact_dataset.py
```

The generator calls the Hugging Face datasets server, preserves no more than
three complete tests and 12 KiB of test input/output per task, and updates
`envs/code_rl/data/source.json` with the selected source and file hashes.
