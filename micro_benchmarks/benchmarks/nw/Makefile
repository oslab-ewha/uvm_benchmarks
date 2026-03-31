#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets -I../../common

default:
	$(NVCC) $(NVCC_FLAGS) ${KERNEL_DIM} needle.cu -lcudart -o needle
	$(NVCC) $(NVCC_FLAGS) ${KERNEL_DIM} needle_memover.cu -lcudart -o needle_memover
	$(NVCC) $(NVCC_FLAGS) ${KERNEL_DIM} needle_memunder.cu -lcudart -o needle_memunder

prefetch:
	$(NVCC) $(NVCC_FLAGS) ${KERNEL_DIM} -DPREF needle.cu -lcudart -o needle

clean:
	rm -f needle needle_memover needle_memunder *.txt gpgpusim_power_report* _cuobjdump_*
