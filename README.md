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
