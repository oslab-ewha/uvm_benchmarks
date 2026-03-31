#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets -I../../common

default:
	$(NVCC) $(NVCC_FLAGS) pathfinder.cu -lcudart -o pathfinder
	$(NVCC) $(NVCC_FLAGS) pathfinder_memunder.cu -lcudart -o pathfinder_memunder
	$(NVCC) $(NVCC_FLAGS) pathfinder_memover.cu -lcudart -o pathfinder_memover

prefetch:
	$(NVCC) $(NVCC_FLAGS) -DPREF pathfinder.cu -lcudart -o pathfinder
	$(NVCC) $(NVCC_FLAGS) -DPREF pathfinder_memunder.cu -lcudart -o pathfinder_memunder
	$(NVCC) $(NVCC_FLAGS) -DPREF pathfinder_memover.cu -lcudart -o pathfinder_memover

clean:
	rm -f pathfinder pathfinder_memunder pathfinder_memover *.txt gpgpusim_power_report* _cuobjdump_*
