# Build llama.cpp locally

The main product of this project is the `llama` library. Its C-style interface can be found in [include/llama.h](../include/llama.h).

The project also includes many example programs and tools using the `llama` library. The examples range from simple, minimal code snippets to sophisticated sub-projects such as an OpenAI-compatible HTTP server.

**To get the Code:**

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

The following sections describe how to build with different backends and options.

## CPU Build

Build llama.cpp using `CMake`:

```bash
cmake -B build
cmake --build build --config Release
```

**Notes**:

- For faster compilation, add the `-j` argument to run multiple jobs in parallel, or use a generator that does this automatically such as Ninja. For example, `cmake --build build --config Release -j 8` will run 8 jobs in parallel.
- For faster repeated compilation, install [ccache](https://ccache.dev/)
- For debug builds, there are two cases:

    1. Single-config generators (e.g. default = `Unix Makefiles`; note that they just ignore the `--config` flag):

       ```bash
       cmake -B build -DCMAKE_BUILD_TYPE=Debug
       cmake --build build
       ```

    2. Multi-config generators (`-G` param set to Visual Studio, XCode...):

       ```bash
       cmake -B build -G "Xcode"
       cmake --build build --config Debug
       ```

    For more details and a list of supported generators, see the [CMake documentation](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html).
- For static builds, add `-DBUILD_SHARED_LIBS=OFF`:
  ```
  cmake -B build -DBUILD_SHARED_LIBS=OFF
  cmake --build build --config Release
  ```

- Building for Windows (x86, x64 and arm64) with MSVC or clang as compilers:
    - Install Visual Studio 2022, e.g. via the [Community Edition](https://visualstudio.microsoft.com/vs/community/). In the installer, select at least the following options (this also automatically installs the required additional tools like CMake,...):
    - Tab Workload: Desktop-development with C++
    - Tab Components (select quickly via search): C++-_CMake_ Tools for Windows, _Git_ for Windows, C++-_Clang_ Compiler for Windows, MS-Build Support for LLVM-Toolset (clang)
    - Please remember to always use a Developer Command Prompt / PowerShell for VS2022 for git, build, test
    - For Windows on ARM (arm64, WoA) build with:
    ```bash
    cmake --preset arm64-windows-llvm-release -D GGML_OPENMP=OFF
    cmake --build build-arm64-windows-llvm-release
    ```
    For building with ninja generator and clang compiler as default:
      -set path:set LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\lib\x64\uwp;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64
      ```bash
      cmake --preset x64-windows-llvm-release
      cmake --build build-x64-windows-llvm-release
      ```
- Curl usage is enabled by default and can be turned off with `-DLLAMA_CURL=OFF`. Otherwise you need to install development libraries for libcurl.
  - **Debian / Ubuntu:** `sudo apt-get install libcurl4-openssl-dev`  # (or `libcurl4-gnutls-dev` if you prefer GnuTLS)
  - **Fedora / RHEL / Rocky / Alma:** `sudo dnf install libcurl-devel`
  - **Arch / Manjaro:** `sudo pacman -S curl`  # includes libcurl headers

## BLAS Build

Building the program with BLAS support may lead to some performance improvements in prompt processing using batch sizes higher than 32 (the default is 512). Using BLAS doesn't affect the generation performance. There are currently several different BLAS implementations available for build and use:

### Accelerate Framework

This is only available on Mac PCs and it's enabled by default. You can just build using the normal instructions.

### OpenBLAS

This provides BLAS acceleration using only the CPU. Make sure to have OpenBLAS installed on your machine.

- Using `CMake` on Linux:

    ```bash
    cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
    cmake --build build --config Release
    ```

### BLIS

Check [BLIS.md](./backend/BLIS.md) for more information.

### Intel oneMKL

Building through oneAPI compilers will make avx_vnni instruction set available for intel processors that do not support avx512 and avx512_vnni. Please note that this build config **does not support Intel GPU**. For Intel GPU support, please refer to [llama.cpp for SYCL](./backend/SYCL.md).

- Using manual oneAPI installation:
  By default, `GGML_BLAS_VENDOR` is set to `Generic`, so if you already sourced intel environment script and assign `-DGGML_BLAS=ON` in cmake, the mkl version of Blas will automatically been selected. Otherwise please install oneAPI and follow the below steps:
    ```bash
    source /opt/intel/oneapi/setvars.sh # You can skip this step if  in oneapi-basekit docker image, only required for manual installation
    cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Intel10_64lp -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_NATIVE=ON
    cmake --build build --config Release
    ```

- Using oneAPI docker image:
  If you do not want to source the environment vars and install oneAPI manually, you can also build the code using intel docker container: [oneAPI-basekit](https://hub.docker.com/r/intel/oneapi-basekit). Then, you can use the commands given above.

Check [Optimizing and Running LLaMA2 on Intel® CPU](https://www.intel.com/content/www/us/en/content-details/791610/optimizing-and-running-llama2-on-intel-cpu.html) for more information.

### Other BLAS libraries

Any other BLAS library can be used by setting the `GGML_BLAS_VENDOR` option. See the [CMake documentation](https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors) for a list of supported vendors.

## Metal Build

On MacOS, Metal is enabled by default. Using Metal makes the computation run on the GPU.
To disable the Metal build at compile time use the `-DGGML_METAL=OFF` cmake option.

When built with Metal support, you can explicitly disable GPU inference with the `--n-gpu-layers 0` command-line argument.

## SYCL

SYCL is a higher-level programming model to improve programming productivity on various hardware accelerators.

llama.cpp based on SYCL is used to **support Intel GPU** (Data Center Max series, Flex series, Arc series, Built-in GPU and iGPU).

For detailed info, please refer to [llama.cpp for SYCL](./backend/SYCL.md).

## CUDA

This provides GPU acceleration using an NVIDIA GPU. Make sure to have the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) installed.

#### Download directly from NVIDIA
You may find the official downloads here: [NVIDIA developer site](https://developer.nvidia.com/cuda-downloads).


#### Compile and run inside a Fedora Toolbox Container
We also have a [guide](./backend/CUDA-FEDORA.md) for setting up CUDA toolkit in a Fedora [toolbox container](https://containertoolbx.org/).

**Recommended for:**
- ***Necessary*** for users of [Atomic Desktops for Fedora](https://fedoraproject.org/atomic-desktops/); such as: [Silverblue](https://fedoraproject.org/atomic-desktops/silverblue/) and [Kinoite](https://fedoraproject.org/atomic-desktops/kinoite/).
  - (there are no supported CUDA packages for these systems)
- ***Necessary*** for users that have a host that is not a: [Supported Nvidia CUDA Release Platform](https://developer.nvidia.com/cuda-downloads).
  - (for example, you may have [Fedora 42 Beta](https://fedoramagazine.org/announcing-fedora-linux-42-beta/) as your your host operating system)
- ***Convenient*** For those running [Fedora Workstation](https://fedoraproject.org/workstation/) or [Fedora KDE Plasma Desktop](https://fedoraproject.org/spins/kde), and want to keep their host system clean.
- *Optionally* toolbox packages are available: [Arch Linux](https://archlinux.org/), [Red Hat Enterprise Linux >= 8.5](https://www.redhat.com/en/technologies/linux-platforms/enterprise-linux), or [Ubuntu](https://ubuntu.com/download)


### Compilation
```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

### Override Compute Capability Specifications

If `nvcc` cannot detect your gpu, you may get compile-warnings such as:
 ```text
nvcc warning : Cannot find valid GPU for '-arch=native', default arch is used
```

To override the `native` GPU detection:

#### 1. Take note of the `Compute Capability` of your NVIDIA devices: ["CUDA: Your GPU Compute > Capability"](https://developer.nvidia.com/cuda-gpus).

```text
GeForce RTX 4090      8.9
GeForce RTX 3080 Ti   8.6
GeForce RTX 3070      8.6
```

#### 2. Manually list each varying `Compute Capability` in the `CMAKE_CUDA_ARCHITECTURES` list.

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;89"
```

### Overriding the CUDA Version

If you have multiple CUDA installations on your system and want to compile llama.cpp for a specific one, e.g. for CUDA 11.7 installed under `/opt/cuda-11.7`:

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/opt/cuda-11.7/bin/nvcc -DCMAKE_INSTALL_RPATH="/opt/cuda-11.7/lib64;\$ORIGIN" -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
```

#### Fixing Compatibility Issues with Old CUDA and New glibc

If you try to use an old CUDA version (e.g. v11.7) with a new glibc version you can get errors like this:

```
/usr/include/bits/mathcalls.h(83): error: exception specification is
  incompatible with that of previous function "cospi"


  /opt/cuda-11.7/bin/../targets/x86_64-linux/include/crt/math_functions.h(5545):
  here
```

It seems the least bad solution is to patch the CUDA installation to declare the correct signatures.
Replace the following lines in `/path/to/your/cuda/installation/targets/x86_64-linux/include/crt/math_functions.h`:

```C++
// original lines
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 cospi(double x);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  cospif(float x);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 sinpi(double x);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  sinpif(float x);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 rsqrt(double x);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  rsqrtf(float x);

// edited lines
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 cospi(double x) noexcept (true);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  cospif(float x) noexcept (true);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 sinpi(double x) noexcept (true);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  sinpif(float x) noexcept (true);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 rsqrt(double x) noexcept (true);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  rsqrtf(float x) noexcept (true);
```

### Runtime CUDA environmental variables

You may set the [cuda environmental variables](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars) at runtime.

```bash
# Use `CUDA_VISIBLE_DEVICES` to hide the first compute device.
CUDA_VISIBLE_DEVICES="-0" ./build/bin/llama-server --model /srv/models/llama.gguf
```

### Unified Memory

The environment variable `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` can be used to enable unified memory in Linux. This allows swapping to system RAM instead of crashing when the GPU VRAM is exhausted. In Windows this setting is available in the NVIDIA control panel as `System Memory Fallback`.

### Performance Tuning

The following compilation options are also available to tweak performance:

| Option                        | Legal values           | Default | Description                                                                                                                                                                                                                                                                                                                                                                      |
|-------------------------------|------------------------|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GGML_CUDA_FORCE_MMQ           | Boolean                | false   | Force the use of custom matrix multiplication kernels for quantized models instead of FP16 cuBLAS even if there is no int8 tensor core implementation available (affects V100, CDNA and RDNA3+). MMQ kernels are enabled by default on GPUs with int8 tensor core support. With MMQ force enabled, speed for large batch sizes will be worse but VRAM consumption will be lower. |
| GGML_CUDA_FORCE_CUBLAS        | Boolean                | false   | Force the use of FP16 cuBLAS instead of custom matrix multiplication kernels for quantized models. There may be issues with numerical overflows (except for CDNA and RDNA4) and memory use will be higher. Prompt processing may become faster on recent datacenter GPUs (the custom kernels were tuned primarily for RTX 3000/4000).                                            |
| GGML_CUDA_PEER_MAX_BATCH_SIZE | Positive integer       | 128     | Maximum batch size for which to enable peer access between multiple GPUs. Peer access requires either Linux or NVLink. When using NVLink enabling peer access for larger batch sizes is potentially beneficial.                                                                                                                                                                  |
| GGML_CUDA_FA_ALL_QUANTS       | Boolean                | false   | Compile support for all KV cache quantization type (combinations) for the FlashAttention CUDA kernels. More fine-grained control over KV cache size but compilation takes much longer.                                                                                                                                                                                           |

## MUSA

This provides GPU acceleration using a Moore Threads GPU. Make sure to have the [MUSA SDK](https://developer.mthreads.com/musa/musa-sdk) installed.

#### Download directly from Moore Threads

You may find the official downloads here: [Moore Threads developer site](https://developer.mthreads.com/sdk/download/musa).

### Compilation

```bash
cmake -B build -DGGML_MUSA=ON
cmake --build build --config Release
```

#### Override Compute Capability Specifications

By default, all supported compute capabilities are enabled. To customize this behavior, you can specify the `MUSA_ARCHITECTURES` option in the CMake command:

```bash
cmake -B build -DGGML_MUSA=ON -DMUSA_ARCHITECTURES="21"
cmake --build build --config Release
```

This configuration enables only compute capability `2.1` (MTT S80) during compilation, which can help reduce compilation time.

#### Compilation options

Most of the compilation options available for CUDA should also be available for MUSA, though they haven't been thoroughly tested yet.

- For static builds, add `-DBUILD_SHARED_LIBS=OFF` and `-DCMAKE_POSITION_INDEPENDENT_CODE=ON`:
  ```
  cmake -B build -DGGML_MUSA=ON \
    -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  cmake --build build --config Release
  ```

### Runtime MUSA environmental variables

You may set the [musa environmental variables](https://docs.mthreads.com/musa-sdk/musa-sdk-doc-online/programming_guide/Z%E9%99%84%E5%BD%95/) at runtime.

```bash
# Use `MUSA_VISIBLE_DEVICES` to hide the first compute device.
MUSA_VISIBLE_DEVICES="-0" ./build/bin/llama-server --model /srv/models/llama.gguf
```

### Unified Memory

The environment variable `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` can be used to enable unified memory in Linux. This allows swapping to system RAM instead of crashing when the GPU VRAM is exhausted.

## HIP

This provides GPU acceleration on HIP-supported AMD GPUs.
Make sure to have ROCm installed.
You can download it from your Linux distro's package manager or from here: [ROCm Quick Start (Linux)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html#rocm-install-quick).

- Using `CMake` for Linux (assuming a gfx1030-compatible AMD GPU):
  ```bash
  HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
      cmake -S . -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release \
      && cmake --build build --config Release -- -j 16
  ```

  Note: `GPU_TARGETS` is optional, omitting it will build the code for all GPUs in the current system.

  To enhance flash attention performance on RDNA3+ or CDNA architectures, you can utilize the rocWMMA library by enabling the `-DGGML_HIP_ROCWMMA_FATTN=ON` option. This requires rocWMMA headers to be installed on the build system.

  The rocWMMA library is included by default when installing the ROCm SDK using the `rocm` meta package provided by AMD. Alternatively, if you are not using the meta package, you can install the library using the `rocwmma-dev` or `rocwmma-devel` package, depending on your system's package manager.

  As an alternative, you can manually install the library by cloning it from the official [GitHub repository](https://github.com/ROCm/rocWMMA), checkout the corresponding version tag (e.g. `rocm-6.2.4`) and set `-DCMAKE_CXX_FLAGS="-I<path/to/rocwmma>/library/include/"` in CMake. This also works under Windows despite not officially supported by AMD.

  Note that if you get the following error:
  ```
  clang: error: cannot find ROCm device library; provide its path via '--rocm-path' or '--rocm-device-lib-path', or pass '-nogpulib' to build without ROCm device library
  ```
  Try searching for a directory under `HIP_PATH` that contains the file
  `oclc_abi_version_400.bc`. Then, add the following to the start of the
  command: `HIP_DEVICE_LIB_PATH=<directory-you-just-found>`, so something
  like:
  ```bash
  HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -p)" \
  HIP_DEVICE_LIB_PATH=<directory-you-just-found> \
      cmake -S . -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release \
      && cmake --build build -- -j 16
  ```

- Using `CMake` for Windows (using x64 Native Tools Command Prompt for VS, and assuming a gfx1100-compatible AMD GPU):
  ```bash
  set PATH=%HIP_PATH%\bin;%PATH%
  cmake -S . -B build -G Ninja -DGPU_TARGETS=gfx1100 -DGGML_HIP=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
  cmake --build build
  ```
  If necessary, adapt `GPU_TARGETS` to the GPU arch you want to compile for. The above example uses `gfx1100` that corresponds to Radeon RX 7900XTX/XT/GRE. You can find a list of targets [here](https://llvm.org/docs/AMDGPUUsage.html#processors)
  Find your gpu version string by matching the most significant version information from `rocminfo | grep gfx | head -1 | awk '{print $2}'` with the list of processors, e.g. `gfx1035` maps to `gfx1030`.


The environment variable [`HIP_VISIBLE_DEVICES`](https://rocm.docs.amd.com/en/latest/understand/gpu_isolation.html#hip-visible-devices) can be used to specify which GPU(s) will be used.
If your GPU is not officially supported you can use the environment variable [`HSA_OVERRIDE_GFX_VERSION`] set to a similar GPU, for example 10.3.0 on RDNA2 (e.g. gfx1030, gfx1031, or gfx1035) or 11.0.0 on RDNA3.

### Unified Memory

On Linux it is possible to use unified memory architecture (UMA) to share main memory between the CPU and integrated GPU by setting environment variable `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`. However, this hurts performance for non-integrated GPUs (but enables working with integrated GPUs).

## Vulkan

### For Windows Users:
**w64devkit**

Download and extract [`w64devkit`](https://github.com/skeeto/w64devkit/releases).

Download and install the [`Vulkan SDK`](https://vulkan.lunarg.com/sdk/home#windows) with the default settings.

Launch `w64devkit.exe` and run the following commands to copy Vulkan dependencies:
```sh
SDK_VERSION=1.3.283.0
cp /VulkanSDK/$SDK_VERSION/Bin/glslc.exe $W64DEVKIT_HOME/bin/
cp /VulkanSDK/$SDK_VERSION/Lib/vulkan-1.lib $W64DEVKIT_HOME/x86_64-w64-mingw32/lib/
cp -r /VulkanSDK/$SDK_VERSION/Include/* $W64DEVKIT_HOME/x86_64-w64-mingw32/include/
cat > $W64DEVKIT_HOME/x86_64-w64-mingw32/lib/pkgconfig/vulkan.pc <<EOF
Name: Vulkan-Loader
Description: Vulkan Loader
Version: $SDK_VERSION
Libs: -lvulkan-1
EOF

```

Switch into the `llama.cpp` directory and build using CMake.
```sh
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release
```

**Git Bash MINGW64**

Download and install [`Git-SCM`](https://git-scm.com/downloads/win) with the default settings

Download and install [`Visual Studio Community Edition`](https://visualstudio.microsoft.com/) and make sure you select `C++`

Download and install [`CMake`](https://cmake.org/download/) with the default settings

Download and install the [`Vulkan SDK`](https://vulkan.lunarg.com/sdk/home#windows) with the default settings.

Go into your `llama.cpp` directory and right click, select `Open Git Bash Here` and then run the following commands

```
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release
```

Now you can load the model in conversation mode using `Vulkan`

```sh
build/bin/Release/llama-cli -m "[PATH TO MODEL]" -ngl 100 -c 16384 -t 10 -n -2 -cnv
```

**MSYS2**

Install [MSYS2](https://www.msys2.org/) and then run the following commands in a UCRT terminal to install dependencies.
```sh
pacman -S git \
    mingw-w64-ucrt-x86_64-gcc \
    mingw-w64-ucrt-x86_64-cmake \
    mingw-w64-ucrt-x86_64-vulkan-devel \
    mingw-w64-ucrt-x86_64-shaderc
```

Switch into the `llama.cpp` directory and build using CMake.
```sh
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release
```

### For Docker users:

You don't need to install the Vulkan SDK. It will be installed inside the container.

```sh
# Build the image
docker build -t llama-cpp-vulkan --target light -f .devops/vulkan.Dockerfile .

# Then, use it:
docker run -it --rm -v "$(pwd):/app:Z" --device /dev/dri/renderD128:/dev/dri/renderD128 --device /dev/dri/card1:/dev/dri/card1 llama-cpp-vulkan -m "/app/models/YOUR_MODEL_FILE" -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 33
```

### For Linux users:

#### Using the LunarG Vulkan SDK

First, follow the official LunarG instructions for the installation and setup of the Vulkan SDK in the [Getting Started with the Linux Tarball Vulkan SDK](https://vulkan.lunarg.com/doc/sdk/latest/linux/getting_started.html) guide.

> [!IMPORTANT]
> After completing the first step, ensure that you have used the `source` command on the `setup_env.sh` file inside of the Vulkan SDK in your current terminal session. Otherwise, the build won't work. Additionally, if you close out of your terminal, you must perform this step again if you intend to perform a build. However, there are ways to make this persistent. Refer to the Vulkan SDK guide linked in the first step for more information about any of this.

#### Using system packages

On Debian / Ubuntu, you can install the required dependencies using:
```sh
sudo apt-get install libvulkan-dev glslc
```

#### Common steps

Second, after verifying that you have followed all of the SDK installation/setup steps, use this command to make sure before proceeding:
```bash
vulkaninfo
```

Then, assuming you have `cd` into your llama.cpp folder and there are no errors with running `vulkaninfo`, you can proceed to build llama.cpp using the CMake commands below:
```bash
cmake -B build -DGGML_VULKAN=1
cmake --build build --config Release
```

Finally, after finishing your build, you should be able to do something like this:
```bash
# Test the output binary
# "-ngl 99" should offload all of the layers to GPU for most (if not all) models.
./build/bin/llama-cli -m "PATH_TO_MODEL" -p "Hi you how are you" -ngl 99

# You should see in the output, ggml_vulkan detected your GPU. For example:
# ggml_vulkan: Using Intel(R) Graphics (ADL GT2) | uma: 1 | fp16: 1 | warp size: 32
```

## CANN
This provides NPU acceleration using the AI cores of your Ascend NPU. And [CANN](https://www.hiascend.com/en/software/cann) is a hierarchical APIs to help you to quickly build AI applications and service based on Ascend NPU.

For more information about Ascend NPU in [Ascend Community](https://www.hiascend.com/en/).

Make sure to have the CANN toolkit installed. You can download it from here: [CANN Toolkit](https://www.hiascend.com/developer/download/community/result?module=cann)

Go to `llama.cpp` directory and build using CMake.
```bash
cmake -B build -DGGML_CANN=on -DCMAKE_BUILD_TYPE=release
cmake --build build --config release
```

You can test with:

```bash
./build/bin/llama-cli -m PATH_TO_MODEL -p "Building a website can be done in 10 steps:" -ngl 32
```

If the following info is output on screen, you are using `llama.cpp` with the CANN backend:
```bash
llm_load_tensors:       CANN model buffer size = 13313.00 MiB
llama_new_context_with_model:       CANN compute buffer size =  1260.81 MiB
```

For detailed info, such as model/device supports, CANN install, please refer to [llama.cpp for CANN](./backend/CANN.md).

## GCU (Enflame TopsRider)

llama.cpp can run on Enflame GCU devices (e.g. S60) via the `ggml-gcu` backend. The backend wraps Enflame's high-level operator library (`topsaten`) and the device runtime (`topsrt`).

### Prerequisites

Install Enflame's TopsRider SDK packages on the build host (Debian/Ubuntu):

- `tops-sdk` provides the runtime headers and `libtopsrt.so` under `/opt/tops`.
- `topsaten` provides the operator library at `/usr/lib/libtopsaten.so` and headers under `/opt/tops/include/gcu/topsaten`.

Verify the install before configuring:

```bash
ls /opt/tops/include/gcu/topsaten/topsaten.h   # headers
ls /usr/lib/libtopsaten.so                     # operator library
ls /opt/tops/lib/libtopsrt.so                  # device runtime
```

### Build

```bash
cmake -B build -DGGML_GCU=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

If the SDK is installed to a non-standard path, override with `-DTOPS_INSTALL_DIR=/path/to/tops`.

### Verify

```bash
./build/bin/llama-cli --list-devices         # GCU0 should appear with its memory size
./build/bin/test-backend-gcu                 # smoke test: ADD + MUL_MAT + RMS_NORM + SOFT_MAX + ROPE + SILU + mixed-dtype MUL_MAT vs CPU
./build/bin/test-backend-ops -b GCU0 -o ADD  # exercises ADD across many shapes/dtypes

# Real model (KV cache stays on CPU via -nkvo):
./build/bin/llama-cli -m model.gguf --device GCU0 -nkvo -p "Hello" -n 16
```

### Performance (Qwen 2.5 0.5B-Instruct, F16, S60, 3 reps)

Measured with `llama-bench`, `--device GCU0 -nkvo 1` versus `-ngl 0` baseline:

| Test       | F16 CPU      | F16 GCU         | Speedup |
|------------|-------------:|----------------:|--------:|
| pp32       |  387.93 t/s  |   676.33 t/s    | 1.74×   |
| pp128      |  467.72 t/s  |  1480.73 t/s    | 3.17×   |
| pp512      |  463.27 t/s  |  1491.79 t/s    | 3.22×   |
| tg16       |   30.51 t/s  |    35.05 t/s    | 1.15×   |
| tg64       |   30.47 t/s  |    34.37 t/s    | 1.13×   |

Prompt processing speeds up cleanly on GCU once the batch is large enough to amortize launch overhead (~3.2× from pp128 onward). Generation gain is small because per-token decode does H↔D copies for the KV cache (`-nkvo` keeps the cache on CPU). MVP-3 perf items: native KV cache offload, native quantized matmul (Q4 weights currently stay on CPU), pinned host buffers, async copy/compute overlap.

For Q4 weights with MVP-3a's dequant-on-load path, GCU runs Q4_K_M at ~31 t/s tg32 / ~309 t/s pp64 — improvement over MVP-2's Q4 GCU (25 / 273) but still below Q4 CPU baseline (70 / 463) because llama.cpp's CPU Q4 kernels are very well tuned and our F16 dequant doubles weight memory plus pays per-token H↔D for the KV cache. Hitting parity with Q4 CPU needs native quantized matmul (`topsatenLinearQuant`) and KV cache offload to GCU — both are tracked for follow-up MVPs.

### Real-model verification

Tested on the S60 against the current backend. All correctness checks pass; performance is model-size-dependent.

**Output equivalence (greedy `--temp 0`)** — same first 16 tokens on both backends:

| Model | CPU output | GCU output (`-nkvo`) | Match |
|---|---|---|---|
| Qwen 2.5 0.5B-Instruct F16 | "The capital of France is Paris." | "The capital of France is Paris." | ✓ |
| Qwen 2.5 0.5B-Instruct Q4_K_M | "The capital of France is Paris." | "The capital of France is Paris." | ✓ |

**F16 Qwen 0.5B at varied context lengths** (3 reps, GCU vs CPU):

| | pp64 | pp128 | pp512 | pp1024 | pp2048 | tg64 | tg128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU | 388 | 468 | 463 | 364 | 321 | 30 | 30 |
| GCU | 676 | 1481 | 1492 | 1073 | 787 | 36 | 35 |
| Speedup | 1.74× | 3.17× | 3.22× | 2.95× | 2.45× | 1.20× | 1.18× |

GCU prompt-processing wins at every tested length, peaking around pp128–512 (~3.2×) and degrading past pp1024 as the F16 KV cache crosses H↔D for every token (`-nkvo` keeps it CPU-resident). Generation is consistently 1.2× faster on GCU.

**Partial offload** (`-ngl 12` of 24 layers): pp64 436 t/s — between CPU-only (388) and full-GCU (676), no crashes. Mixed CPU+GCU layer split works as expected.

**Architecture coverage:** verified loading and running on Qwen 2 (Qwen 2.5 0.5B) and Llama (TinyLlama 15M Q4_0, Llama 3.2 1.2B Q4_K_M). Backend isn't Qwen-specific.

**Q4 model size scaling — the crossover** (where Q4 GCU starts beating Q4 CPU) is between 0.5B and 1.2B params:

| Q4_K_M Model | pp64 CPU | pp64 GCU | Speedup |
|---|---:|---:|---:|
| Qwen 2.5 0.5B (~0.63B params) | 463 t/s | 309 t/s | 0.67× (CPU wins) |
| Llama 3.2 1B (~1.24B params) | 240 t/s | **576 t/s** | **2.40× (GCU wins)** |

A 3.6× swing from doubling the model size: GCU's per-op launch overhead amortizes over larger per-layer compute. Generation throughput is similar on both backends for both models (~30 t/s) because per-token decode is bounded by H↔D for the KV cache (`-nkvo` keeps it CPU-resident). For real models in this size range and up, **GCU is the clear win for prompt-heavy workloads** (RAG, summarization). Generation-heavy workloads see modest gains.

**Long-context durability** — GCU advantage holds at long contexts on the 1B model:

| Llama 1B Q4_K_M | pp64 | pp128 | pp512 | pp1024 | pp2048 | pp4096 |
|---|---:|---:|---:|---:|---:|---:|
| CPU | 240 | 248 | 236 | 221 | 196 | 162 |
| GCU | 576 | 700 | 645 | 539 | 408 | 288 |
| Speedup | 2.40× | 2.83× | 2.73× | 2.44× | 2.08× | 1.78× |

The advantage degrades as context grows (the F16 KV cache crossing H↔D for every token bottlenecks both backends), but remains substantial at 4K context. Larger models would show even more durable advantage.

**Recommended invocation for real models on GCU:**
```sh
./build/bin/llama-cli -m model.gguf --device GCU0 -nkvo -p "..." -n 64
```

Use F16 GGUFs for the best perf. Q4 GGUFs work but currently lose to highly-tuned Q4 CPU kernels (see "Known SDK ceilings" below).

### Operator coverage (MVP-2)

**29 ops dispatched / 22 distinct kernels** (the 4 view ops are zero-copy and 3 of `CPY`/`DUP`/`CONT` share one handler; 7 unary subtypes share the `GGML_OP_UNARY` dispatcher and 4 GLU subtypes share the `GGML_OP_GLU` dispatcher but each calls distinct topsaten kernels). Any op or shape outside these is automatically routed to CPU by ggml's scheduler:

- Element-wise: `ADD`, `MUL`, `SCALE` (bias = 0 only), `CONCAT`
- Activations: `SILU`, `GELU`, `GELU_QUICK`, `RELU`, `TANH`, `SIGMOID`, `HARDSWISH`, `HARDSIGMOID` (all via `GGML_OP_UNARY`)
- Gated activations: `GEGLU`, `GEGLU_QUICK`, `SWIGLU`, `REGLU` (all via `GGML_OP_GLU`; both two-source and split forms)
- Normalization: `NORM` (LayerNorm without affine), `RMS_NORM`
- Position encoding: `ROPE` (NORMAL mode 0 + NEOX mode 2; no YARN, no MROPE/VISION/IMROPE; F32 and F16)
- Reduction: `SOFT_MAX` (with optional mask, `max_bias = 0`, no softmax sinks)
- Linear: `MUL_MAT` (F32×F32→F32 fast path; F16-weight × {F16,F32} → {F16,F32} via cast; Q4_0 / Q8_0 / Q4_K weights via F16 dequant-on-load)
- MoE dispatch: `MUL_MAT_ID` (same dtype matrix as `MUL_MAT`; per-(token, expert-slot) `topsatenLinear` loop; F16-weight path casts F32 input once for the whole sweep). Required for Gemma 4 26B A4B and other MoE models.
- Indexing: `GET_ROWS` (F32 only, unbatched), `SET_ROWS` (F32 dst only — KV cache stays on CPU)
- Memory: `CPY`/`DUP`/`CONT` (same-dtype contiguous + F32↔F16 conversion); view ops (`RESHAPE`, `VIEW`, `PERMUTE`, `TRANSPOSE`)

### Known limitations (MVP-2)

- Q4_0, Q8_0, and Q4_K weight tensors are dequantized to F16 at model-load time (one-time host cost) and stored as F16 on GCU (2-4× the on-disk size). MUL_MAT then runs on GCU via the F16 path. Other Q-types (Q5_K, Q6_K, Q3_K, etc.) stay on CPU. Native quantized matmul via `topsatenLinearQuant` is a future MVP that would avoid the F16 expansion and likely match Q4 CPU performance.
- KV cache is designed to stay on CPU. `SET_ROWS` to F16 destinations (the cache dtype) is refused on GCU. Pass `-nkvo` (`--no-kv-offload`) when offloading layers to GCU; without it llama.cpp tries to allocate the cache on GCU and the scheduler aborts at graph_reserve. With `-nkvo`, real Q4 / F16 models load and run on `--device GCU0` (Q4 weights and KV stay CPU; activation math runs on GCU). An MVP-3b probe (manual D2D memcpy bypassing `topsatenIndexPut`) was 2-5× slower than `-nkvo` because per-call sync H2D of indices drains the stream — native cache offload needs an async index transfer or a custom GCU scatter kernel.
- BF16 not supported.
- Only `ROPE` mode 0 is implemented; YARN / NEOX / MROPE go to CPU.
- `SOFT_MAX` with `max_bias != 0` (alibi) and `softmax sinks` (a non-null `op->src[2]`) go to CPU.
- Single device, single stream. Pinned host memory (`topsHostMalloc`) is enabled and used by the `-nkvo` KV cache. On Llama 3.2 1B Q4_K_M (`--device GCU0 -nkvo 1`, r=5):

  | test  | pinned (default) | no-pinned         | uplift |
  |-------|------------------|-------------------|--------|
  | tg32  | 33.80 ± 0.23 t/s | 32.15 ± 0.09 t/s  | +5.1%  |
  | tg64  | 34.27 ± 0.19 t/s | 31.12 ± 0.20 t/s  | +10.1% |
  | pp512 | 677.34 ± 5.04    | 643.86 ± 8.24     | +5.2%  |

  Set `GGML_GCU_NO_PINNED=1` to fall back to a normal CPU buffer.

  Async H↔D / compute overlap (`topsMemcpyAsync` on a dedicated copy stream, event-mediated handoff with the compute stream — MVP-4a) stacks on top of pinned memory. Llama 3.2 1B Q4_K_M (`--device GCU0 -nkvo 1`, r=5):

  | test  | MVP-4a active        | MVP-4a disabled      | uplift |
  |-------|----------------------|----------------------|--------|
  | tg32  | 30.95 ± 0.29 t/s     | 30.93 ± 0.12 t/s     | +0.1%  |
  | tg64  | 31.36 ± 0.10 t/s     | 31.05 ± 0.16 t/s     | +1.0%  |
  | pp512 | 672.52 ± 16.13 t/s   | 677.81 ± 17.76 t/s   | -0.8%  |

  Qwen 2.5 0.5B F16 (`--device GCU0 -nkvo 1`, r=5):

  | test  | MVP-4a active        | MVP-4a disabled      | uplift |
  |-------|----------------------|----------------------|--------|
  | tg32  | 33.50 ± 0.63 t/s     | 31.28 ± 1.14 t/s     | +7.1%  |
  | tg64  | 32.83 ± 0.75 t/s     | 31.58 ± 0.73 t/s     | +4.0%  |
  | pp512 | 1650.18 ± 39.71 t/s  | 1669.72 ± 3.15 t/s   | -1.2%  |

  Set `GGML_GCU_NO_ASYNC_COPY=1` to fall back to synchronous `topsMemcpy`. Multi-device support is MVP-5 work.

  **Honest read of MVP-4a's payoff.** Diagnostic instrumentation confirmed that ggml's scheduler does **not** route cross-backend tensor copies (the per-layer KV reads under `-nkvo`) through `set_tensor_async` / `get_tensor_async`; those always use the synchronous buffer-level path (`buffer.set_tensor` / `get_tensor`). The only consumer of the async backend interface in this stack is llama.cpp's sampler fetching the logit tensor once per decode step. On a 32-token Qwen 0.5B F16 run we measured 9 async-get calls / ~5 MiB total — a transfer share of ~0.08 % of decode wall-clock, far below the +7 % uplift in the table above. That uplift is therefore at most ~1.7 σ of session-to-session noise; the Llama Q4_K_M flat result is mechanically forced (zero async calls ever fire). MVP-4a is correctly wired and adds no overhead, so it stays on by default — but real per-token overlap will only materialize once either the ggml scheduler is taught to use the async backend interface for cross-backend copies, or the GCU backend's own per-op `topsStreamSynchronize` is removed (MVP-4b — Scope-2).

  **MVP-4b — queued ops (drop per-op compute sync).** Replaces every op handler's `topsStreamSynchronize` + `pool.free` with a deferred-free routed through `gcu_release_scratch`. Kernels now queue on `compute_stream` without per-op host round-trips; scratch returns to the pool at `graph_compute` end after a single drain synchronize. Llama 3.2 1B Q4_K_M (`--device GCU0 -nkvo 1`, r=5):

  | test  | MVP-4b active       | MVP-4b disabled     | uplift |
  |-------|---------------------|---------------------|--------|
  | tg32  | 34.05 ± 0.35 t/s    | 28.57 ± 0.44 t/s    | +19.2% |
  | tg64  | 33.76 ± 0.15 t/s    | 27.86 ± 0.36 t/s    | +21.2% |
  | pp512 | 631.98 ± 13.36 t/s  | 605.25 ± 15.75 t/s  | +4.4%  |

  Qwen 2.5 0.5B F16 (`--device GCU0 -nkvo 1`, r=5):

  | test  | MVP-4b active        | MVP-4b disabled      | uplift |
  |-------|----------------------|----------------------|--------|
  | tg32  | 43.37 ± 0.08 t/s     | 26.91 ± 0.66 t/s     | +61.2% |
  | tg64  | 41.36 ± 2.57 t/s     | 26.80 ± 0.38 t/s     | +54.3% |
  | pp512 | 1634.06 ± 43.89 t/s  | 1597.57 ± 15.68 t/s  | +2.3%  |

  Set `GGML_GCU_NO_QUEUED_OPS=1` to revert each call site to the pre-MVP-4b sync-and-free pattern.

  Gemma 4 26B A4B (MoE, 25.23B total / 3.8B active per token, Q4_K_M, `--device GCU0 -nkvo 1`, r=2):

  | test  | MVP-4b active       | MVP-4b disabled     | uplift  |
  |-------|---------------------|---------------------|---------|
  | tg16  | 7.65 ± 0.23 t/s     | 6.11 ± 0.04 t/s     | +25.2%  |
  | pp64  | 35.68 ± 1.79 t/s    | 38.30 ± 2.24 t/s    | within noise |

  This is the first MoE workload to validate against the new `MUL_MAT_ID` op and the queued-ops path together. The tg uplift (+25%) is meaningful: the per-token compute amortizes launch overhead more than a pure dense decode would, so the MVP-4b benefit is bounded but real. pp64 difference is within the stddev band (single run pair); a longer r=5 sweep would tighten the prefill comparison.

  **Honest read of MVP-4b's payoff.** The token-generation uplift is large and real: +19–21 % on Llama 1B Q4_K_M, +54–61 % on Qwen 0.5B F16, and +25 % on Gemma 4 26B A4B — well outside session-to-session noise in all three. This directly confirms that per-op `topsStreamSynchronize` host round-trips were the dominant tg bottleneck, not compute throughput itself. Prefill (pp512) gains are smaller (+4 % / +2 %) because its runtime is dominated by matrix multiply kernel time, not host-side dispatch overhead; the relative overhead of a per-op sync is diluted over longer per-op GPU work. The Qwen F16 tg uplift is larger than Llama Q4_K_M's because F16 decode ops are individually cheaper (less compute per op) so the fixed host-roundtrip cost is a larger fraction of total time. In practice MVP-4b brings GCU tg throughput from the ~28–34 t/s range (CPU-synchronized level) to the ~34–43 t/s range — a meaningful step, though absolute throughput is still gated by the S60's per-kernel launch latency at this scale.

### Known SDK ceilings (per-topsaten investigation)

A few proposed perf improvements were explored against the topsop SDK source (`/Users/root1/gitlab/topsop`) and turned out to require GCU device-kernel work (writing `.tops` code), not just topsaten/topsrt API wiring. Documenting here so future contributors don't re-investigate:

- **Native quantized MUL_MAT** via `topsatenLinearQuant`: blocked by format mismatch. The op only accepts **int8** weights with group_size of −1, 64, or 128; ggml stores Q4_0/Q4_K as **4-bit** with block_size 32 (Q4_0) or 256-element super-blocks (Q4_K). Bridging the two needs either offline re-quantization to a topsaten-friendly layout (changes GGUF file format) or a custom `.tops` kernel that consumes ggml's packed nibble layout directly. Source check: `op_aten_linearquant.h:132-220`.
- **Native KV cache offload (F16 SET_ROWS)** via `topsatenIndexPut`: rejected at runtime for the actual cache shape. Manual D2D memcpy bypass works correctness-wise but per-call sync H2D of indices serialized the layer pipeline and ran 2-5× slower than `-nkvo`. A real win needs a custom `.tops` scatter kernel or asynchronous batched indexing.
- **Block-paged KV cache** via `topsvllmReshapeAndCache`: requires the vLLM `[num_blocks, block_size, num_heads, head_size]` layout. ggml uses a flat contiguous cache; switching would mean rewriting llama.cpp's KV management — out of scope for this backend.

The current MVP-2/3a/3c state appears to be the practical ceiling on this SDK without writing GCU device kernels. Further perf gains are real but require a different class of work (`.tops` device code) than this backend takes on.

## ZenDNN

ZenDNN provides optimized deep learning primitives for AMD EPYC™ CPUs. It accelerates matrix multiplication operations for inference workloads.

### Compilation

- Using `CMake` on Linux (automatic build):

    ```bash
    cmake -B build -DGGML_ZENDNN=ON
    cmake --build build --config Release
    ```

    The first build will automatically download and build ZenDNN, which may take 5-10 minutes. Subsequent builds will be much faster.

- Using `CMake` with custom ZenDNN installation:

    ```bash
    cmake -B build -DGGML_ZENDNN=ON -DZENDNN_ROOT=/path/to/zendnn/install
    cmake --build build --config Release
    ```

### Testing

You can test with:

```bash
./build/bin/llama-cli -m PATH_TO_MODEL -p "Building a website can be done in 10 steps:" -n 50
```

For detailed information about hardware support, setup instructions, and performance optimization, refer to [llama.cpp for ZenDNN](./backend/ZenDNN.md).

## Arm® KleidiAI™
KleidiAI is a library of optimized microkernels for AI workloads, specifically designed for Arm CPUs. These microkernels enhance performance and can be enabled for use by the CPU backend.

To enable KleidiAI, go to the llama.cpp directory and build using CMake
```bash
cmake -B build -DGGML_CPU_KLEIDIAI=ON
cmake --build build --config Release
```
You can verify that KleidiAI is being used by running
```bash
./build/bin/llama-cli -m PATH_TO_MODEL -p "What is a car?"
```
If KleidiAI is enabled, the ouput will contain a line similar to:
```
load_tensors: CPU_KLEIDIAI model buffer size =  3474.00 MiB
```
KleidiAI's microkernels implement optimized tensor operations using Arm CPU features such as dotprod, int8mm and SME. llama.cpp selects the most efficient kernel based on runtime CPU feature detection. However, on platforms that support SME, you must manually enable SME microkernels by setting the environment variable `GGML_KLEIDIAI_SME=1`.

Depending on your build target, other higher priority backends may be enabled by default. To ensure the CPU backend is used, you must disable the higher priority backends either at compile time, e.g. -DGGML_METAL=OFF, or during run-time using the command line option `--device none`.

## OpenCL

This provides GPU acceleration through OpenCL on recent Adreno GPU.
More information about OpenCL backend can be found in [OPENCL.md](./backend/OPENCL.md) for more information.

### Android

Assume NDK is available in `$ANDROID_NDK`. First, install OpenCL headers and ICD loader library if not available,

```sh
mkdir -p ~/dev/llm
cd ~/dev/llm

git clone https://github.com/KhronosGroup/OpenCL-Headers && \
cd OpenCL-Headers && \
cp -r CL $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include

cd ~/dev/llm

git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader && \
cd OpenCL-ICD-Loader && \
mkdir build_ndk && cd build_ndk && \
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DOPENCL_ICD_LOADER_HEADERS_DIR=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=24 \
  -DANDROID_STL=c++_shared && \
ninja && \
cp libOpenCL.so $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android
```

Then build llama.cpp with OpenCL enabled,

```sh
cd ~/dev/llm

git clone https://github.com/ggml-org/llama.cpp && \
cd llama.cpp && \
mkdir build-android && cd build-android

cmake .. -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_OPENCL=ON

ninja
```

### Windows Arm64

First, install OpenCL headers and ICD loader library if not available,

```powershell
mkdir -p ~/dev/llm

cd ~/dev/llm
git clone https://github.com/KhronosGroup/OpenCL-Headers && cd OpenCL-Headers
mkdir build && cd build
cmake .. -G Ninja `
  -DBUILD_TESTING=OFF `
  -DOPENCL_HEADERS_BUILD_TESTING=OFF `
  -DOPENCL_HEADERS_BUILD_CXX_TESTS=OFF `
  -DCMAKE_INSTALL_PREFIX="$HOME/dev/llm/opencl"
cmake --build . --target install

cd ~/dev/llm
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader && cd OpenCL-ICD-Loader
mkdir build && cd build
cmake .. -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_PREFIX_PATH="$HOME/dev/llm/opencl" `
  -DCMAKE_INSTALL_PREFIX="$HOME/dev/llm/opencl"
cmake --build . --target install
```

Then build llama.cpp with OpenCL enabled,

```powershell
cmake .. -G Ninja `
  -DCMAKE_TOOLCHAIN_FILE="$HOME/dev/llm/llama.cpp/cmake/arm64-windows-llvm.cmake" `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_PREFIX_PATH="$HOME/dev/llm/opencl" `
  -DBUILD_SHARED_LIBS=OFF `
  -DGGML_OPENCL=ON
ninja
```

## Android

To read documentation for how to build on Android, [click here](./android.md)

## WebGPU [In Progress]

The WebGPU backend relies on [Dawn](https://dawn.googlesource.com/dawn). Follow the instructions [here](https://dawn.googlesource.com/dawn/+/refs/heads/main/docs/quickstart-cmake.md) to install Dawn locally so that llama.cpp can find it using CMake. The currrent implementation is up-to-date with Dawn commit `bed1a61`.

In the llama.cpp directory, build with CMake:

```
cmake -B build -DGGML_WEBGPU=ON
cmake --build build --config Release
```

### Browser Support

WebGPU allows cross-platform access to the GPU from supported browsers. We utilize [Emscripten](https://emscripten.org/) to compile ggml's WebGPU backend to WebAssembly. Emscripten does not officially support WebGPU bindings yet, but Dawn currently maintains its own WebGPU bindings called emdawnwebgpu.

Follow the instructions [here](https://dawn.googlesource.com/dawn/+/refs/heads/main/src/emdawnwebgpu/) to download or build the emdawnwebgpu package (Note that it might be safer to build the emdawbwebgpu package locally, so that it stays in sync with the version of Dawn you have installed above). When building using CMake, the path to the emdawnwebgpu port file needs to be set with the flag `EMDAWNWEBGPU_DIR`.

## IBM Z & LinuxONE

To read documentation for how to build on IBM Z & LinuxONE, [click here](./build-s390x.md)

## Notes about GPU-accelerated backends

The GPU may still be used to accelerate some parts of the computation even when using the `-ngl 0` option. You can fully disable GPU acceleration by using `--device none`.

In most cases, it is possible to build and use multiple backends at the same time. For example, you can build llama.cpp with both CUDA and Vulkan support by using the `-DGGML_CUDA=ON -DGGML_VULKAN=ON` options with CMake. At runtime, you can specify which backend devices to use with the `--device` option. To see a list of available devices, use the `--list-devices` option.

Backends can be built as dynamic libraries that can be loaded dynamically at runtime. This allows you to use the same llama.cpp binary on different machines with different GPUs. To enable this feature, use the `GGML_BACKEND_DL` option when building.
