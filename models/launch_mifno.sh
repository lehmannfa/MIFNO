#!/bin/bash

# Load the environment
# TO DO: Adapt the modules to your computing centre

NLAYERS=16 # total number of layers
DV="16"
LIST_D1=""
LIST_D2=""
LIST_D3=""
LIST_M1=""
LIST_M2=""
LIST_M3=""

i=5 # the first 4 layers keep the same dimensions
while [ $i -le $NLAYERS ]
do
    LIST_D1+="32 "
    LIST_D2+="32 "
    LIST_D3+="32 "
    LIST_M1+="16 "
    LIST_M2+="16 "
    LIST_M3+="16 "

    i=$(( $i + 1 ))
done

LIST_D1+="32 32 32 32"
LIST_D2+="32 32 32 32"
LIST_D3+="64 128 256 320"
LIST_M1+="16 16 16 16"
LIST_M2+="16 16 16 16"
LIST_M3+="16 32 32 32"

ccc_mprun python3 train.py @model_type "MIFNO" @nlayers ${NLAYERS} @branching_index 4 @dv ${DV} @list_D1 ${LIST_D1} @list_D2 ${LIST_D2} @list_D3 ${LIST_D3} @list_M1 ${LIST_M1} @list_M2 ${LIST_M2} @list_M3 ${LIST_M3} @learning_rate 0.0004 @Ntrain 27000 @Nval 3000 @batch_size 16 @source_orientation "moment" @normalize_source @normalize_traces @epochs 200 @dir_data_train "../data/formatted/HEMEWS3D_S32_Z32_T320_fmax5_rot0_train" @dir_data_val "../data/formatted/HEMEWS3D_S32_Z32_T320_fmax5_rot0_val" @dir_logs "../logs/"
