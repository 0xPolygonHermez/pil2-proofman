# Pil2-stark Lib

## Compiling locally

Steps to compile `pil2-stark` locally:
### Clone repository

```sh
git clone --recursive https://github.com/0xPolygonHermez/pil2-stark.git
cd pil2-stark
```

### Install dependencies

The following packages must be installed.

#### Ubuntu/Debian

```sh
apt update
apt install build-essential libbenchmark-dev libomp-dev libgmp-dev nlohmann-json3-dev nasm libsodium-dev cmake
```

#### openSUSE
```sh
zypper addrepo https://download.opensuse.org/repositories/network:cryptocurrencies/openSUSE_Tumbleweed/network:cryptocurrencies.repo
zypper refresh
zypper install -t pattern devel_basis
zypper install libbenchmark1 libomp16-devel libgmp10 nlohmann_json-devel nasm libsodium-devel cmake
```

#### Fedora
```
dnf group install "C Development Tools and Libraries" "Development Tools"
dnf config-manager --add-repo https://terra.fyralabs.com/terra.repo
dnf install google-benchmark-devel libomp-devel gmp gmp-devel gmp-c++ nlohmann-json-devel nasm libsodium-devel cmake
```

### Compilation

Run `make` to compile the main project:

```sh
make clean
make generate
make starks_lib -j
```

### GPU Compilation

Requires CUDA Toolkit installed. The GPU library is built separately:

```sh
make starks_lib_gpu -j
```

By default, the build auto-detects the architecture of the host GPU. To target specific architectures, use `CUDA_GENCODE_FLAGS`:

```sh
# Single architecture (faster build)
make starks_lib_gpu -j CUDA_GENCODE_FLAGS="-gencode arch=compute_89,code=sm_89 -gencode arch=compute_89,code=compute_89"

# Multiple architectures
make starks_lib_gpu -j CUDA_GENCODE_FLAGS="-gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90"
```

When building via Cargo (pil2-proofman), use the `CUDA_ARCHS` environment variable instead — it generates the flags automatically:

```sh
# Default: auto-detects host GPU architecture (same as direct make)
cargo build --release --features gpu

# All major architectures (distribution build): sm_80, sm_86, sm_89, sm_90, sm_100 + PTX, sm_120 + PTX
CUDA_ARCHS="major" cargo build --release --features gpu

# Single architecture
CUDA_ARCHS="89" cargo build --release --features gpu

# Multiple architectures
CUDA_ARCHS="89,90" cargo build --release --features gpu
```

To inspect which architectures are embedded in the compiled library:

```sh
cuobjdump -all lib-gpu/libstarksgpu.a 2>/dev/null | grep "arch ="
```


