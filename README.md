# UVM prefetch experiments

### ■ NVIDIA `Unified Virtual Memory` Kernel Module
Original Source codes
- https://github.com/NVIDIA/open-gpu-kernel-modules/tree/545.23/kernel-open/nvidia-uvm


### ■ Micro benchmarks
References
1. https://github.com/DebashisGanguly/gpgpu-sim_UVMSmart
2. https://github.com/cavazos-lab/PolyBench-ACC
3. https://github.com/yuhc/gpu-rodinia


|                **Benchmark**                | **Ref.** |
|:-------------------------------------------:|:--------:|
|         2D.Convolution<br/>(2DCONV)         |    [1]   |
|             $A^{T}AX$<br/>(ATAX)            |    [1]   |
|        Breadth-First Search<br/>(bfs)       |    [1]   |
|       Biconjugate Gradient<br/>(bicg)       |    [2]   |
| Finite-Difference Time-Domain<br/>(FDTD-2D) | [1], [2] |
|          Needleman-Wunsch<br/>(nw)          |    [1]   |
|             Pathfinder<br/>(pf)             | [1], [3] |
