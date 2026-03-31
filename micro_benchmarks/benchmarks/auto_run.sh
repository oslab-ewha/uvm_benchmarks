#!/bin/bash

PROP_MOD="/data"${HOME}"/test_uvm/nvidia-uvm.ko"
#ORGN_MOD="/lib/modules/5.4.0-100-generic/kernel/drivers/video/nvidia-uvm.ko"
ORGN_MOD="/lib/modules/5.4.0-216-generic/updates/dkms/nvidia-uvm.ko"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS=("ATAX" "2DCONV" "FDTD-2D" "bicg" "nw" "pathfinder") #"bfs"
THRESHOLDS=(51 1 100)

experiments() {
    for i in {1..3}
    do
        start=$(date +%s%6N)    # start_time (epoch sec + micro sec)

        ./run    # Benchmarks

        end=$(date +%s%6N)    # end_time

        elapsed=$((end - start))
        echo "Elapsed time: ${elapsed} µs"
        #echo "Elapsed time: $((elapsed/1000000)) s"
    done
}

read -s -p "Password: " password

for DIR in "${BENCHMARKS[@]}"; do
    cd "${BASE_DIR}/${DIR}" || {
        echo "Directory ${BASE_DIR}/${DIR} not found, skip..."
        continue
    }

    # Cold start
    #./run

    # Proposed
    echo "$password" | sudo -S sh -c "rmmod nvidia-uvm"
    echo "$password" | sudo -S sh -c "insmod "$PROP_MOD" \
            uvm_perf_prefetch_enable=1 uvm_uXuA_printk=0 uvm_perf_prefetch_threshold=1"
    experiments

    # Conventional
    for T in "${THRESHOLDS[@]}"; do
        echo "$password" | sudo -S sh -c "rmmod nvidia-uvm"
        echo "$password" | sudo -S sh -c "insmod "$ORGN_MOD" \
                uvm_perf_prefetch_threshold=${T}"
        experiments
    done
    
    cd "${BASE_DIR}"
done

