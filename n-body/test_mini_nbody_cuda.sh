#!/bin/bash

script_path="./mini-nbody/cuda"
pushd ${script_path} > /dev/null
echo "#---Test mini-nbody CUDA direct translation---#"
./shmoo-cuda-nbody-orig.sh
echo "#---Test mini-nbody CUDA +SOA optimization---#"
./shmoo-cuda-nbody-soa.sh
echo "#---Test mini-nbody CUDA +FTZ optimization---#"
./shmoo-cuda-nbody-ftz.sh
echo "#---Test mini-nbody CUDA +block loading optimization---#"
./shmoo-cuda-nbody-block.sh
echo "#---Test mini-nbody CUDA +unroll optimization---#"
./shmoo-cuda-nbody-unroll.sh
popd > /dev/null
