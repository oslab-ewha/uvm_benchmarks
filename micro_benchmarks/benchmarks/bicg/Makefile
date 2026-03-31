#!/usr/bin/make

NVCC_FLAGS = -g -Wno-deprecated-gpu-targets -I../../common

default:
	nvcc $(NVCC_FLAGS) bicg.cu -lcudart -o bicg
	nvcc $(NVCC_FLAGS) bicg_memunder.cu -lcudart -o bicg_memunder
	nvcc $(NVCC_FLAGS) bicg_memover.cu -lcudart -o bicg_memover

clean:
	rm -rf bicg bicg_memunder bicg_memover gpgpusim_power_report* _cuobjdump_* *.txt