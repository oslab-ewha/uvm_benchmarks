#!/usr/bin/make

NVCC_FLAGS = -g -Wno-deprecated-gpu-targets -I../../common

default:
	nvcc $(NVCC_FLAGS) atax.cu -lcudart -o atax
	nvcc $(NVCC_FLAGS) atax_memover.cu -lcudart -o atax_memover
	nvcc $(NVCC_FLAGS) atax_memunder.cu -lcudart -o atax_memunder

clean:
	rm -rf atax atax_2d atax_memover atax_memunder gpgpusim_power_report* _cuobjdump_* *.txt
