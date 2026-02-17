# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Development build (fastest, native GPU only)
cmake -B build
cmake --build build -j$(nproc)

# Run
./build/LichtFeld-Studio -d data/test

# Portable/distribution build (self-contained)
cmake -B build -DBUILD_PORTABLE=ON
cmake --build build -j$(nproc)
cmake --install build --prefix ./dist
./dist/bin/run_lichtfeld.sh -d data/test
```

**Requirements**: CMake 3.30+, CUDA 12.8+, GCC 14+ (Linux), vcpkg (`VCPKG_ROOT` env var)

**Key build options**:
- `BUILD_PORTABLE=ON` - Self-contained distribution with bundled libs
- `BUILD_TESTS=ON` - Enable test suite
- `BUILD_CUDA_MIN_SM=75` - Minimum GPU (75=Turing, 80=Ampere, 89=Ada)

## Testing

```bash
# Build and run all tests
cmake -B build -DBUILD_TESTS=ON
cmake --build build
ctest --test-dir build

# Run specific test
ctest --test-dir build -R TestName
```

## Code Formatting

clang-format is enforced. Install pre-commit hook:
```bash
cp tools/pre-commit .git/hooks/
```

Manual formatting:
```bash
clang-format -i path/to/file.cpp
```

## Python Development

**Always use `uv` as the Python virtual environment manager, not `pip` directly.**

```bash
# For Python scripts in this repo
cd scripts/360pipeline
uv sync                           # Install dependencies
uv run python ../script.py        # Run scripts
uv add package-name               # Add new dependency (not pip install)
```

## Architecture

LichtFeld Studio is a high-performance 3D Gaussian Splatting implementation in C++23/CUDA.

**Module hierarchy** (dependencies flow downward):
```
app (entry point)
  └─ visualizer (lfs::vis) - ImGui UI, OpenGL rendering, scene management
      └─ rendering (lfs::rendering) - Rasterization pipeline
          └─ training (lfs::training) - GS optimization, MCMC, loss kernels
              └─ core (lfs::core) - Custom tensor library, CUDA utilities
```

**Key modules**:
- **core**: Custom tensor implementation (no LibTorch), CUDA kernels, memory management
- **training**: Gaussian optimization, multiple rasterization backends (FastGS, GSplat), MCMC strategy
- **rendering**: Real-time OpenGL rasterization
- **visualizer**: Interactive viewer, timeline/sequencer, gizmos, property system
- **io**: Format loaders (PLY, NeRF, COLMAP, transforms.json), video I/O via FFmpeg
- **python**: nanobind-based Python bindings

**Namespace convention**: `lfs::core::`, `lfs::training::`, `lfs::rendering::`, `lfs::vis::`, `lfs::io::`

## File Headers

New files require SPDX headers:
```cpp
/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
```

## 360 Video Pipeline

For equirectangular 360 video to 3DGS:
```bash
cd scripts/360pipeline
uv sync
uv run python ../360_to_3dgs.py input.mp4 output/ --iterations 10000
```

Requires: ffmpeg, OpenSfM (for SfM). Uses `--gut` flag for equirectangular support.
