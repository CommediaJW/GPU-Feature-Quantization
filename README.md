# GPU-Feature-Quantization

## Install 
Requirement:
* CUDA >= 11.3

Install python dependencies.
```shell
$ pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116
```

Install the system packages for building the shared library.
```shell
$ sudo apt-get update
$ sudo apt-get install -y build-essential python3-dev make cmake
```

Download the source files.
```shell
$ git clone git@github.com:CommediaJW/GPU-Feature-Quantization.git
```

Build.
```shell
$ mkdir build && cd build
$ cmake ..
$ make -j16
```

After building, it will generate `libbifeat.so` in `${WORKSPACE}/build`.