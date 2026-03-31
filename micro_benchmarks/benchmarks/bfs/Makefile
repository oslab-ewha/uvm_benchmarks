#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets -I../../common

default:
	$(NVCC) $(NVCC_FLAGS) bfs.cu -lcudart -o bfs 
	$(NVCC) $(NVCC_FLAGS) bfs_memover.cu -lcudart -o bfs_memover
	$(NVCC) $(NVCC_FLAGS) bfs_memunder.cu -lcudart -o bfs_memunder

prefetch:
	$(NVCC) $(NVCC_FLAGS) -DPREF bfs.cu -lcudart -o bfs 

clean:
	rm -f bfs bfs_memover bfs_memunder *.linkinfo *.txt gpgpusim_power_report* _cuobjdump_*
