#!/usr/bin/make

NVCC_FLAGS = -g -Wno-deprecated-gpu-targets -I../../common

default:
	nvcc $(NVCC_FLAGS) 2DConvolution.cu -lcudart -o 2dconv
	nvcc $(NVCC_FLAGS) 2DConv_memunder.cu -lcudart -o 2dconv_memunder
	nvcc $(NVCC_FLAGS) 2DConv_memover.cu -lcudart -o 2dconv_memover

clean:
	rm -rf 2dconv 2dconv_memunder 2dconv_memover gpgpusim_power_report* _cuobjdump_* *.txt
