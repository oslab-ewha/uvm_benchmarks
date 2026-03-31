#!/usr/bin/make

NVCC_FLAGS = -g -Wno-deprecated-gpu-targets -I../../common

default:
	nvcc $(NVCC_FLAGS) fdtd2d.cu -lcudart -o fdtd2d
	nvcc $(NVCC_FLAGS) fdtd2d_memunder.cu -lcudart -o fdtd2d_memunder
	nvcc $(NVCC_FLAGS) fdtd2d_memover.cu -lcudart -o fdtd2d_memover

clean:
	rm -rf fdtd2d fdtd2d_memunder fdtd2d_memover gpgpusim_power_report* _cuobjdump_* *.txt
