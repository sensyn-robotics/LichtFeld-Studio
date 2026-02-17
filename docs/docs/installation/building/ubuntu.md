# Building on Ubuntu 24.04

This guide covers building LichtFeld Studio from source on Ubuntu 24.04 LTS.

## Requirements

- Ubuntu 24.04 LTS
- NVIDIA GPU (Turing or newer recommended)
- At least 16GB RAM (32GB recommended for large scenes)
- ~20GB disk space for build

## 1. Install System Dependencies

```bash
# Update package lists
sudo apt update

# Install essential build tools
sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    pkg-config \
    ninja-build \
    autoconf \
    automake \
    libtool \
    zip \
    unzip \
    tar

# Install libraries required by vcpkg packages
sudo apt install -y \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxxf86vm-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libbz2-dev \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    libraw-dev \
    libopenjp2-7-dev \
    libssl-dev \
    libffi-dev \
    libsqlite3-dev \
    zlib1g-dev \
    liblzma-dev \
    libreadline-dev \
    nasm \
    yasm
```

## 2. Install CMake 3.30+

Ubuntu 24.04 ships with CMake 3.28, but we need 3.30+. Install from Kitware's official APT repository:

```bash
# Remove old cmake if installed
sudo apt remove --purge cmake

# Install Kitware's signing key
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
    gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null

# Add Kitware repository
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' | \
    sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null

# Install cmake
sudo apt update
sudo apt install -y cmake

# Verify version (should be 3.30+)
cmake --version
```

## 3. Install GCC 14

Ubuntu 24.04 ships with GCC 13, but we need GCC 14+:

```bash
# Add Ubuntu Toolchain PPA
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt update

# Install GCC 14
sudo apt install -y gcc-14 g++-14

# Set as default compiler
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 100

# Verify version
gcc --version
g++ --version
```

## 4. Install CUDA Toolkit 12.8+

```bash
# Download and install CUDA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA Toolkit (this will install the latest version)
sudo apt install -y cuda-toolkit

# Add CUDA to PATH (add to ~/.bashrc for persistence)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify installation
nvcc --version
```

> **Note**: You may need to reboot after CUDA installation for the NVIDIA driver to load properly.

## 5. Install vcpkg

```bash
# Clone vcpkg
git clone https://github.com/Microsoft/vcpkg.git ~/vcpkg

# Bootstrap vcpkg
cd ~/vcpkg
./bootstrap-vcpkg.sh

# Set VCPKG_ROOT environment variable (add to ~/.bashrc for persistence)
export VCPKG_ROOT=~/vcpkg
```

Add to `~/.bashrc`:
```bash
echo 'export VCPKG_ROOT=~/vcpkg' >> ~/.bashrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## 6. Build LichtFeld Studio

```bash
# Clone the repository
git clone https://github.com/sensyn-robotics/LichtFeld-Studio.git
cd LichtFeld-Studio

# Configure (first run will install vcpkg dependencies - takes 10-30 minutes)
cmake -B build

# Build (adjust -j based on your CPU cores)
cmake --build build -j$(nproc)

# Test the build
./build/LichtFeld-Studio --help
```

## 7. (Optional) Install sccache for Faster Rebuilds

```bash
# Install sccache
cargo install sccache
# Or download prebuilt binary from https://github.com/mozilla/sccache/releases

# CMake will automatically detect and use sccache
```

## Troubleshooting

### CMake can't find CUDA
```bash
# Ensure CUDA is in PATH
export PATH=/usr/local/cuda/bin:$PATH
# Re-run cmake configuration
cmake -B build --fresh
```

### vcpkg dependency build failures
```bash
# Clean vcpkg build cache and retry
rm -rf ~/vcpkg/buildtrees
rm -rf ~/vcpkg/packages
cmake -B build --fresh
```

### "no kernel image is available" at runtime
Your GPU is older than the default minimum SM. Rebuild with a lower SM version:
```bash
cmake -B build -DBUILD_CUDA_MIN_SM=70  # For Volta GPUs
cmake --build build -j$(nproc)
```

### OpenGL errors
```bash
# Install additional OpenGL packages
sudo apt install -y libglvnd-dev libegl1-mesa-dev
```

---

# 360 Video Pipeline Setup

For using the 360 video to 3DGS pipeline (`scripts/360_to_3dgs.py`):

## Install ffmpeg

```bash
sudo apt install -y ffmpeg
```

## Install uv (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

## Install OpenSfM

OpenSfM is required for SfM reconstruction from equirectangular images:

```bash
# Install OpenSfM dependencies
sudo apt install -y \
    python3-dev \
    python3-pip \
    libboost-all-dev \
    libeigen3-dev \
    libceres-dev \
    libgflags-dev \
    libgoogle-glog-dev

# Clone and install OpenSfM
git clone --recursive https://github.com/mapillary/OpenSfM.git
cd OpenSfM

# Create virtual environment and install
python3 -m venv venv
source venv/bin/activate
pip install -e .

# Add opensfm to PATH (or activate venv when using)
export PATH=$PATH:$(pwd)/bin
```

## Run the Pipeline

```bash
cd LichtFeld-Studio/scripts/360pipeline

# Sync dependencies
uv sync

# Run the pipeline
uv run python ../360_to_3dgs.py /path/to/360video.mp4 /path/to/output --iterations 10000
```

## Pipeline Options

```
--fps FLOAT         Frame extraction rate (default: 1.0)
--iterations INT    Training iterations (default: 10000)
--start FLOAT       Start time in seconds
--duration FLOAT    Duration in seconds
--skip-extraction   Use existing frames
--skip-sfm          Use existing reconstruction
--skip-training     Only run SfM (no 3DGS training)
```
