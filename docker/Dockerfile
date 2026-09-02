# gl-serve for NVIDIA aarch64 machines — one image, several GPUs.
#
# Verified targets:
#   Jetson Orin Nano Super   Ampere   sm_87    8 GB shared
#   DGX Spark (GB10)         Blackwell sm_121  ~120 GB unified
#
# Both are aarch64, so this is ONE native image, not a cross-build. The
# GPUs differ, so the CUDA objects are a fatbinary covering both — the
# same trick the Modal deployment used for T4+L4 ("75;89"). sm_121 needs
# CUDA 13.x; older toolkits do not know it.
#
# THE RULE THAT MAKES THIS PORTABLE: the CUDA *runtime* ships in the
# image, the *driver* does not. libcuda.so.1 differs completely between
# Orin's Tegra nvgpu stack and Spark's Blackwell driver, and is injected
# from the host at run time by CDI (`--device nvidia.com/gpu=all`).
# Baking a driver in produces an image that works on exactly one machine.
# gl-serve is linked against the CUDA stubs for precisely this reason, so
# `libcuda.so.1 => not found` inside the image is correct, not a fault.
#
# Weights are NOT baked in. They are 5.6 GB, they change rarely, and the
# code changes often — bundling them means moving 6+ GB to ship a 20 MB
# binary change. Mount them at /opt/gemma-live/models.

ARG CUDA=13.2.1
ARG CUDA_ARCHS=87;121

# ── build ────────────────────────────────────────────────────────────
FROM --platform=linux/arm64 nvidia/cuda:${CUDA}-devel-ubuntu24.04 AS build
ARG CUDA_ARCHS
ARG JOBS=4
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        git cmake build-essential ninja-build \
        libcurl4-openssl-dev pkg-config libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Source only. Host build directories and virtualenvs must not come
# along: a previous version of this shipped 923 MB of macOS .so files
# into an aarch64 image. The .dockerignore enforces it; this is the
# second line of defence.
COPY . /src
WORKDIR /src

# Belt and braces behind .dockerignore. A stale build directory from the
# host carries a CMakeCache.txt with absolute paths baked in, and cmake
# refuses to reuse it:
#   "The current CMakeCache.txt directory ... is different than the
#    directory ... where CMakeCache.txt was created"
# The ignore file should prevent it arriving at all; this makes the build
# correct even when it does.
RUN find /src -type d \( -name build -o -name 'build-*' -o -name .venv \) \
         -prune -exec rm -rf {} + 2>/dev/null || true

# CrispASR first — gl-serve links libcrispasr-tts.so, and CMakeLists.txt
# hardcodes its Linux location as <CrispASR>/build-static/src, so the
# directory name is load-bearing.
RUN cmake -S vendor/CrispASR -B vendor/CrispASR/build-static \
        -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_SHARED_LINKER_FLAGS=-Wl,--exclude-libs,ALL \
 && cmake --build vendor/CrispASR/build-static --target crispasr-tts -j"${JOBS}"

# -L.../stubs -lcuda: this container has no driver, and libcrispasr-tts
# references the CUDA driver API (cuMem*). Resolved against the real
# libcuda.so.1 that CDI injects at run time.
RUN cmake -S . -B build -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DCMAKE_EXE_LINKER_FLAGS='-L/usr/local/cuda/lib64/stubs -lcuda' \
 && cmake --build build --target gl-serve -j"${JOBS}"

# Collect into the layout the runtime stage expects. $ORIGIN/../lib keeps
# the binary relocatable, replacing the absolute RPATH CMake bakes in.
RUN set -eux; \
    mkdir -p /out/bin /out/lib; \
    cp build/gl-serve /out/bin/; \
    find build vendor/CrispASR/build-static -name '*.so*' -type f -exec cp -a {} /out/lib/ \; ; \
    find build vendor/CrispASR/build-static -name '*.so*' -type l -exec cp -a {} /out/lib/ \; ; \
    cp -a web prompts keywords /out/; \
    [ -d voices ] && cp -a voices /out/ || true; \
    patchelf --version >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y --no-install-recommends patchelf); \
    patchelf --set-rpath '$ORIGIN/../lib' /out/bin/gl-serve; \
    for f in /out/lib/*.so*; do [ -f "$f" ] && patchelf --set-rpath '$ORIGIN' "$f" || true; done

# ── runtime ──────────────────────────────────────────────────────────
FROM --platform=linux/arm64 nvidia/cuda:${CUDA}-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive

# libcublas is not in the runtime image by default; ggml-cuda needs it.
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        libcublas-13-2 libgomp1 libcurl4 openssl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /out /opt/gemma-live

# gl-serve resolves models/, prompts/, keywords/ and web/ RELATIVE TO ITS
# CWD. Run it anywhere else and the readiness probe gets a 500 from a
# missing web root, which reads as "never came up".
WORKDIR /opt/gemma-live
ENV PATH=/opt/gemma-live/bin:$PATH

# Mount the weights here. Deliberately a VOLUME so a run without them
# fails loudly at startup rather than silently serving nothing.
VOLUME ["/opt/gemma-live/models"]
EXPOSE 8927

# --web-host 0.0.0.0: inside a container the port is namespaced, so
# binding loopback would make -p useless. Put a TLS terminator in front
# if it is exposed beyond a trusted network.
ENTRYPOINT ["gl-serve", "--web-host", "0.0.0.0", "--web-port", "8927"]
