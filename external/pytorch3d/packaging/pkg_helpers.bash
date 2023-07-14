# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# shellcheck shell=bash
# A set of useful bash functions for common functionality we need to do in
# many build scripts

# Setup CUDA environment variables, based on CU_VERSION
#
# Inputs:
#   CU_VERSION (cu92, cu100, cu101, cu102)
#   NO_CUDA_PACKAGE (bool)
#   BUILD_TYPE (conda, wheel)
#
# Outputs:
#   VERSION_SUFFIX (e.g., "")
#   PYTORCH_VERSION_SUFFIX (e.g., +cpu)
#   WHEEL_DIR (e.g., cu100/)
#   CUDA_HOME (e.g., /usr/local/cuda-9.2, respected by torch.utils.cpp_extension)
#   FORCE_CUDA (respected by pytorch3d setup.py)
#   NVCC_FLAGS (respected by pytorch3d setup.py)
#
# Precondition: CUDA versions are installed in their conventional locations in
# /usr/local/cuda-*
#
# NOTE: Why VERSION_SUFFIX versus PYTORCH_VERSION_SUFFIX?  If you're building
# a package with CUDA on a platform we support CUDA on, VERSION_SUFFIX ==
# PYTORCH_VERSION_SUFFIX and everyone is happy.  However, if you are building a
# package with only CPU bits (e.g., torchaudio), then VERSION_SUFFIX is always
# empty, but PYTORCH_VERSION_SUFFIX is +cpu (because that's how you get a CPU
# version of a Python package.  But that doesn't apply if you're on OS X,
# since the default CU_VERSION on OS X is cpu.
setup_cuda() {

  # First, compute version suffixes.  By default, assume no version suffixes
  export VERSION_SUFFIX=""
  export PYTORCH_VERSION_SUFFIX=""
  export WHEEL_DIR=""
  # Wheel builds need suffixes (but not if they're on OS X, which never has suffix)
  if [[ "$BUILD_TYPE" == "wheel" ]] && [[ "$(uname)" != Darwin ]]; then
    # The default CUDA has no suffix
    if [[ "$CU_VERSION" != "cu102" ]]; then
      export PYTORCH_VERSION_SUFFIX="+$CU_VERSION"
    fi
    # Match the suffix scheme of pytorch, unless this package does not have
    # CUDA builds (in which case, use default)
    if [[ -z "$NO_CUDA_PACKAGE" ]]; then
      export VERSION_SUFFIX="$PYTORCH_VERSION_SUFFIX"
      export WHEEL_DIR="$CU_VERSION/"
    fi
  fi

  # Now work out the CUDA settings
  case "$CU_VERSION" in
    cu116)
      if [[ "$OSTYPE" == "msys" ]]; then
        export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.6"
      else
        export CUDA_HOME=/usr/local/cuda-11.6/
      fi
      export FORCE_CUDA=1
      # Hard-coding gencode flags is temporary situation until
      # https://github.com/pytorch/pytorch/pull/23408 lands
      export NVCC_FLAGS="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_50,code=compute_50"
      ;;
    cu115)
      if [[ "$OSTYPE" == "msys" ]]; then
        export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5"
      else
        export CUDA_HOME=/usr/local/cuda-11.5/
      fi
      export FORCE_CUDA=1
      # Hard-coding gencode flags is temporary situation until
      # https://github.com/pytorch/pytorch/pull/23408 lands
      export NVCC_FLAGS="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_50,code=compute_50"
      ;;
    cu113)
      if [[ "$OSTYPE" == "msys" ]]; then
        export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.3"
      else
        export CUDA_HOME=/usr/local/cuda-11.3/
      fi
      export FORCE_CUDA=1
      # Hard-coding gencode flags is temporary situation until
      # https://github.com/pytorch/pytorch/pull/23408 lands
      export NVCC_FLAGS="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_50,code=compute_50"
      ;;
    cu112)
      if [[ "$OSTYPE" == "msys" ]]; then
        export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2"
      else
        export CUDA_HOME=/usr/local/cuda-11.2/
      fi
      export FORCE_CUDA=1
      # Hard-coding gencode flags is temporary situation until
      # https://github.com/pytorch/pytorch/pull/23408 lands
      export NVCC_FLAGS="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_50,code=compute_50"
      ;;
    cu111)
      if [[ "$OSTYPE" == "msys" ]]; then
        export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1"
      else
        export CUDA_HOME=/usr/local/cuda-11.1/
      fi
      export FORCE_CUDA=1
      # Hard-coding gencode flags is temporary situation until
      # https://github.com/pytorch/pytorch/pull/23408 lands
      export NVCC_FLAGS="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_50,code=compute_50"
      ;;
    cu110)
      if [[ "$OSTYPE" == "msys" ]]; then
        export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0"
      else
        export CUDA_HOME=/usr/local/cuda-11.0/
      fi
      export FORCE_CUDA=1
      # Hard-coding gencode flags is temporary situation until
      # https://github.com/pytorch/pytorch/pull/23408 lands
      export NVCC_FLAGS="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_50,code=compute_50"
      ;;
    cu102)
      if [[ "$OSTYPE" == "msys" ]]; then
        export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2"
      else
        export CUDA_HOME=/usr/local/cuda-10.2/
      fi
      export FORCE_CUDA=1
      # Hard-coding gencode flags is temporary situation until
      # https://github.com/pytorch/pytorch/pull/23408 lands
      export NVCC_FLAGS="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_50,code=compute_50"
      ;;
    cu101)
      if [[ "$OSTYPE" == "msys" ]]; then
        export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1"
      else
        export CUDA_HOME=/usr/local/cuda-10.1/
      fi
      export FORCE_CUDA=1
      # Hard-coding gencode flags is temporary situation until
      # https://github.com/pytorch/pytorch/pull/23408 lands
      export NVCC_FLAGS="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_50,code=compute_50"
      ;;
    cu100)
      if [[ "$OSTYPE" == "msys" ]]; then
        export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0"
      else
        export CUDA_HOME=/usr/local/cuda-10.0/
      fi
      export FORCE_CUDA=1
      # Hard-coding gencode flags is temporary situation until
      # https://github.com/pytorch/pytorch/pull/23408 lands
      export NVCC_FLAGS="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_50,code=compute_50"
      ;;
    cu92)
      if [[ "$OSTYPE" == "msys" ]]; then
        export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.2"
      else
        export CUDA_HOME=/usr/local/cuda-9.2/
      fi
      export FORCE_CUDA=1
      export NVCC_FLAGS="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_50,code=compute_50"
      ;;
    cpu)
      ;;
    *)
      echo "Unrecognized CU_VERSION=$CU_VERSION"
      exit 1
      ;;
  esac
}

# Populate build version if necessary, and add version suffix
#
# Inputs:
#   BUILD_VERSION (e.g., 0.2.0 or empty)
#   VERSION_SUFFIX (e.g., +cpu)
#
# Outputs:
#   BUILD_VERSION (e.g., 0.2.0.dev20190807+cpu)
#
# Fill BUILD_VERSION if it doesn't exist already with a nightly string
# Usage: setup_build_version 0.2.0
setup_build_version() {
  if [[ -z "$BUILD_VERSION" ]]; then
    export BUILD_VERSION="$1.dev$(date "+%Y%m%d")$VERSION_SUFFIX"
  else
    export BUILD_VERSION="$BUILD_VERSION$VERSION_SUFFIX"
  fi
}

# Set some useful variables for OS X, if applicable
setup_macos() {
  if [[ "$(uname)" == Darwin ]]; then
    export MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++
  fi
}

# Top-level entry point for things every package will need to do
#
# Usage: setup_env 0.2.0
setup_env() {
  setup_cuda
  setup_build_version "$1"
  setup_macos
}

# Function to retry functions that sometimes timeout or have flaky failures
retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

# Inputs:
#   PYTHON_VERSION (2.7, 3.5, 3.6, 3.7)
#   UNICODE_ABI (bool)
#
# Outputs:
#   PATH modified to put correct Python version in PATH
#
# Precondition: If Linux, you are in a soumith/manylinux-cuda* Docker image
setup_wheel_python() {
  if [[ "$(uname)" == Darwin ]]; then
    eval "$(conda shell.bash hook)"
    conda env remove -n "env$PYTHON_VERSION" || true
    conda create -yn "env$PYTHON_VERSION" python="$PYTHON_VERSION"
    conda activate "env$PYTHON_VERSION"
  else
    case "$PYTHON_VERSION" in
      2.7)
        if [[ -n "$UNICODE_ABI" ]]; then
          python_abi=cp27-cp27mu
        else
          python_abi=cp27-cp27m
        fi
        ;;
      3.5) python_abi=cp35-cp35m ;;
      3.6) python_abi=cp36-cp36m ;;
      3.7) python_abi=cp37-cp37m ;;
      3.8) python_abi=cp38-cp38 ;;
      *)
        echo "Unrecognized PYTHON_VERSION=$PYTHON_VERSION"
        exit 1
        ;;
    esac
    export PATH="/opt/python/$python_abi/bin:$PATH"
  fi
}

# Install with pip a bit more robustly than the default
pip_install() {
  retry pip install --progress-bar off "$@"
}

# Install torch with pip, respecting PYTORCH_VERSION, and record the installed
# version into PYTORCH_VERSION, if applicable
setup_pip_pytorch_version() {
  if [[ -z "$PYTORCH_VERSION" ]]; then
    # Install latest prerelease version of torch, per our nightlies, consistent
    # with the requested cuda version
    pip_install --pre torch -f "https://download.pytorch.org/whl/nightly/${WHEEL_DIR}torch_nightly.html"
    if [[ "$CUDA_VERSION" == "cpu" ]]; then
      # CUDA and CPU are ABI compatible on the CPU-only parts, so strip
      # in this case
      export PYTORCH_VERSION="$(pip show torch | grep ^Version: | sed 's/Version:  *//' | sed 's/+.\+//')"
    else
      export PYTORCH_VERSION="$(pip show torch | grep ^Version: | sed 's/Version:  *//')"
    fi
  else
    pip_install "torch==$PYTORCH_VERSION$CUDA_SUFFIX" \
      -f https://download.pytorch.org/whl/torch_stable.html \
      -f https://download.pytorch.org/whl/nightly/torch_nightly.html
  fi
}

# Fill PYTORCH_VERSION with the latest conda nightly version, and
# CONDA_CHANNEL_FLAGS with appropriate flags to retrieve these versions
#
# You MUST have populated CUDA_SUFFIX before hand.
setup_conda_pytorch_constraint() {
  if [[ -z "$PYTORCH_VERSION" ]]; then
    export CONDA_CHANNEL_FLAGS="-c pytorch-nightly"
    export PYTORCH_VERSION="$(conda search --json 'pytorch[channel=pytorch-nightly]' | \
                              python -c "import os, sys, json, re; cuver = os.environ.get('CU_VERSION'); \
                               cuver_1 = cuver.replace('cu', 'cuda') if cuver != 'cpu' else cuver; \
                               cuver_2 = (cuver[:-1] + '.' + cuver[-1]).replace('cu', 'cuda') if cuver != 'cpu' else cuver; \
                               print(re.sub(r'\\+.*$', '', \
                                [x['version'] for x in json.load(sys.stdin)['pytorch'] \
                                  if (x['platform'] == 'darwin' or cuver_1 in x['fn'] or cuver_2 in x['fn']) \
                                    and 'py' + os.environ['PYTHON_VERSION'] in x['fn']][-1]))")"
    if [[ -z "$PYTORCH_VERSION" ]]; then
      echo "PyTorch version auto detection failed"
      echo "No package found for CU_VERSION=$CU_VERSION and PYTHON_VERSION=$PYTHON_VERSION"
      exit 1
    fi
  else
    export CONDA_CHANNEL_FLAGS="-c pytorch"
  fi
  if [[ "$CU_VERSION" == cpu ]]; then
    export CONDA_PYTORCH_BUILD_CONSTRAINT="- pytorch==$PYTORCH_VERSION${PYTORCH_VERSION_SUFFIX}"
    export CONDA_PYTORCH_CONSTRAINT="- pytorch==$PYTORCH_VERSION"
  else
    export CONDA_PYTORCH_BUILD_CONSTRAINT="- pytorch==${PYTORCH_VERSION}${PYTORCH_VERSION_SUFFIX}"
    export CONDA_PYTORCH_CONSTRAINT="- pytorch==${PYTORCH_VERSION}${PYTORCH_VERSION_SUFFIX}"
  fi
  export PYTORCH_VERSION_NODOT=${PYTORCH_VERSION//./}
}

# Translate CUDA_VERSION into CUDA_CUDATOOLKIT_CONSTRAINT
setup_conda_cudatoolkit_constraint() {
  export CONDA_CPUONLY_FEATURE=""
  export CONDA_CUB_CONSTRAINT=""
  if [[ "$(uname)" == Darwin ]]; then
    export CONDA_CUDATOOLKIT_CONSTRAINT=""
  else
    case "$CU_VERSION" in
      cu116)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=11.6,<11.7 # [not osx]"
        ;;
      cu115)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=11.5,<11.6 # [not osx]"
        ;;
      cu113)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=11.3,<11.4 # [not osx]"
        ;;
      cu112)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=11.2,<11.3 # [not osx]"
        ;;
      cu111)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=11.1,<11.2 # [not osx]"
        ;;
      cu110)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=11.0,<11.1 # [not osx]"
        # Even though cudatoolkit 11.0 provides CUB we need our own, to control the
        # version, because the built-in 1.9.9 in the cudatoolkit causes problems.
        export CONDA_CUB_CONSTRAINT="- nvidiacub"
        ;;
      cu102)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=10.2,<10.3 # [not osx]"
        export CONDA_CUB_CONSTRAINT="- nvidiacub"
        ;;
      cu101)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=10.1,<10.2 # [not osx]"
        export CONDA_CUB_CONSTRAINT="- nvidiacub"
        ;;
      cu100)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=10.0,<10.1 # [not osx]"
        export CONDA_CUB_CONSTRAINT="- nvidiacub"
        ;;
      cu92)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=9.2,<9.3 # [not osx]"
        export CONDA_CUB_CONSTRAINT="- nvidiacub"
        ;;
      cpu)
        export CONDA_CUDATOOLKIT_CONSTRAINT=""
        export CONDA_CPUONLY_FEATURE="- cpuonly"
        ;;
      *)
        echo "Unrecognized CU_VERSION=$CU_VERSION"
        exit 1
        ;;
    esac
  fi
}

# Build the proper compiler package before building the final package
setup_visual_studio_constraint() {
  if [[ "$OSTYPE" == "msys" ]]; then
      export VSTOOLCHAIN_PACKAGE=vs2019
      export VSDEVCMD_ARGS=''
      # shellcheck disable=SC2086
      conda build $CONDA_CHANNEL_FLAGS --no-anaconda-upload packaging/$VSTOOLCHAIN_PACKAGE
      cp packaging/$VSTOOLCHAIN_PACKAGE/conda_build_config.yaml packaging/pytorch3d/conda_build_config.yaml
  fi
}

download_nvidiacub_if_needed() {
  case "$CU_VERSION" in
    cu110|cu102|cu101|cu100|cu92)
      echo "Downloading cub"
      wget --no-verbose https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
      tar xzf 1.10.0.tar.gz
      CUB_HOME=$(realpath ./cub-1.10.0)
      export CUB_HOME
      echo "CUB_HOME is now $CUB_HOME"
      ;;
  esac
  # We don't need CUB for a cpu build or if cuda is 11.1 or higher
}
