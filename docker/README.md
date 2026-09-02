# gl-serve in a container

One aarch64 image, several NVIDIA machines. Verified targets:

| | GPU | arch | memory |
|---|---|---|---|
| Jetson Orin Nano Super | Ampere | `sm_87` | 8 GB shared |
| DGX Spark (GB10) | Blackwell | `sm_121` | ~120 GB unified |

Both are aarch64, so this is a native build on either — or on an Apple
Silicon Mac. It is **not** a cross-build, and `build.sh` refuses to run
on x86_64 rather than emulate every `nvcc` invocation.

## Why one image works on two different GPUs

The CUDA objects are a fatbinary built for both architectures
(`-DCMAKE_CUDA_ARCHITECTURES="87;121"`), so the right kernels are
selected at load time. `sm_121` requires CUDA 13.x; older toolkits do not
know it.

More importantly: **the CUDA runtime ships in the image, the driver does
not.** `libcuda.so.1` differs completely between Orin's Tegra `nvgpu`
stack and Spark's Blackwell driver, and CDI injects the host's own at run
time. `gl-serve` is linked against the CUDA stubs for exactly this
reason, so `libcuda.so.1 => not found` *inside* the image is correct.

Bake a driver in and you get an image that runs on precisely one machine.

## Setup, once per host

```bash
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list                     # expect nvidia.com/gpu=all
```

Regenerate the CDI spec after a driver or JetPack upgrade — the spec
pins host library paths and goes stale.

## Build and ship

```bash
./docker/build.sh                       # both archs
CUDA_ARCHS=87 ./docker/build.sh         # Orin only, ~half the time

docker save gemma-live:<tag> | zstd | ssh <host> 'zstd -d | docker load'
./docker/sync-models.sh tbtlr@orin.local
```

Build time is dominated by nvcc on llama.cpp's fattn template instances
and scales with the number of architectures. Memory is the real limit,
not cores: 24 GB for `-j4`, and below that it dies **silently** — no
error, just a dead build, recorded at 39%, 56% and (at 16 GB) 31%. If a
build vanishes without a message, give the VM more memory before
lowering `JOBS`.

## Run

```bash
docker compose up -d
```

Identical on both machines. Per-host differences go in the environment:

```bash
# Orin — 8 GB shared, genuinely near its limit with these weights
GEMMA_LIVE_ARGS=--llm-vision-off docker compose up -d
```

`--llm-vision-off` reclaims the vision tower (~215 MB of weights plus its
compute buffer). Pointless on Spark.

## Things that will bite

**`gl-serve` resolves `models/`, `prompts/`, `keywords/` and
`web/index.html` relative to its CWD.** The image sets `WORKDIR
/opt/gemma-live`; run it elsewhere and the readiness probe gets a 500
from a missing `web/index.html`, which presents as "the backend never
came up".

**Weights are a mount, not a layer.** 5.6 GB that changes rarely, versus
a binary that changes often. `compose.yaml` mounts them read-only, and
the `VOLUME` declaration makes a run without them fail at startup rather
than serve nothing quietly.

**`tokenizer.bin` has no reference anywhere in the repo.** moonshine
resolves it as `dirname(model)/tokenizer.bin`, and without it startup
fails complaining about the *model*. `sync-models.sh` checks for it.

**The healthcheck has a 5-minute `start_period`** because a cold load of
5.6 GB genuinely takes minutes on an Orin. `/` only answers once the
models are loaded, so it is a readiness signal rather than a liveness
ping.

**The container binds `0.0.0.0`**, since a namespaced loopback would make
`-p` useless. Put a TLS terminator in front of it on anything beyond a
trusted network.
