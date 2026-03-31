#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIRS=("2DCONV" "FDTD-2D" "bicg" "nw") #"ATAX" "pathfinder") #,"bfs"

#----path----
PYTHON=${HOME}"/anaconda3/envs/uvm/bin/python"
PROP_MOD="/data"${HOME}"/test_uvm/new_prefetch_2511/nvidia-uvm.ko"
ORGN_MOD="/lib/modules/5.4.0-100-generic/kernel/drivers/video/nvidia-uvm.ko"
#------------

read -s -p "Password: " password
echo "$password" | sudo -S sh -c "rmmod nvidia-uvm"
echo "$password" | sudo -S sh -c "insmod "$PROP_MOD" \
                uvm_perf_prefetch_enable=1 uvm_uXuA_printk=1 uvm_perf_prefetch_threshold=51"

for TARGET_DIR in "${TARGET_DIRS[@]}"; do
    echo "${SCRIPT_DIR}/${TARGET_DIR}"
    cd ${SCRIPT_DIR}/${TARGET_DIR} || exit 1
    LOG_FILE="${SCRIPT_DIR}/process_kernel_logs_${TARGET_DIR}_251209.txt"

    START_TS=$(date '+%Y-%m-%d %H:%M:%S.%N')

    ./run

    END_TS=$(date '+%Y-%m-%d %H:%M:%S.%N')

    echo "$password" | sudo -S journalctl -k \
            --since "$START_TS" \
            --until "$END_TS" \
            --output=short-precise \
            --no-pager > "$LOG_FILE"

done

: << COMMENTS
# Object function from kernel print
# python script
${PYTHON} << EOF

import pandas as pd
import numpy as np

logfile="$LOG_FILE"

df = pd.read_csv(logfile, skiprows=1, header=None, delimiter=':', on_bad_lines='warn')
df[3] = df[3].str.strip()

'''
uXuAb : gpu->id, batch_context->batch_id, batch_context->num_cached_faults, batch_context->num_coalesced_faults,
        : batch_context->num_duplicate_faults, batch_context->num_invalid_prefetch_faults
'''
#df[3] = df[3].str.strip()
#df_b = df[df[3]=='uXuAb']
#df_bs = df_b[4].str.split(',', expand=True)
#df_bs_ = df_bs[df_bs[3].isna()]
#df_bs__ = df_bs[df_bs[3].notna()]
#df_bs_['dec_0'] = df_bs_[0].str.replace(r'00000$', '', regex=True).apply(lambda x: int(x, 16))
#plt.scatter(df_bs_.index, df_bs_['dec_0']); plt.show()
#df_bs__[5] = df_bs__[0].astype(int)/df_bs__[1].astype(int)
#plt.boxplot(df_bs__[5]); plt.show()

'''
uXuAr : src_id, dst_id, address, size
'''
df_r = df.loc[df[3]=='uXuAr']
df_rs = df_r[4].str.split(',', expand=True)
df_rs[2] = df_rs[2].apply(lambda x: int(x, 16))
df_rs[[0,1,3]] = df_rs[[0,1,3]].astype(np.int64)
df_rs_c = df_rs[(df_rs[0]==0) | (df_rs[1]==0)]
df_rs_g = df_rs[(df_rs[0]!=0) & (df_rs[1]!=0)]

pci_trans = (df_rs_c[3].sum()) / (512) * 16.7    # usec
nvlink_trans = (df_rs_g[3].sum()) / (512) * 5.1    # usec

print(pci_trans + nvlink_trans)#, len(df_rs_c), len(df_rs_g))

'''
uXuAe
'''
#df_e = df[df[3]=='uXuAe']

EOF
COMMENTS

