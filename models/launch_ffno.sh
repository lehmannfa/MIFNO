#!/bin/bash

# Load the environment
# TO DO: adapt the modules to your computing centre

NLAYERS=16 # total number of layers
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

# increase the vertical dimension in the last layers
LIST_D1+="32 32 32 32"
LIST_D2+="32 32 32 32"
LIST_D3+="64 128 256 320"
LIST_M1+="16 16 16 16"
LIST_M2+="16 16 16 16"
LIST_M3+="16 32 32 32"

# Increase the number of channels to make things comparable with MIFNO
DV="16"
LIST_DV="${DV} "

i=1 # the first layers keep the same dimensions
while [ $i -le 4 ] # when comparing with MIFNO(branching_index=4)
do
    LIST_DV+="${DV} "
    i=$(( $i + 1 ))
done

DDV=$(( $DV * 3))
while [ $i -le $NLAYERS ] # when comparing with MIFNO(branching_index=4)
do
    LIST_DV+="${DDV} "
    i=$(( $i + 1 ))
done

ccc_mprun python3 train.py @model_type "FFNO" @nlayers ${NLAYERS} @dv ${DV} @list_dv ${LIST_DV} @list_D1 ${LIST_D1} @list_D2 ${LIST_D2} @list_D3 ${LIST_D3} @list_M1 ${LIST_M1} @list_M2 ${LIST_M2} @list_M3 ${LIST_M3} @learning_rate 0.0004 @Ntrain 27000 @Nval 3000 @batch_size 16 @source_orientation "moment" @normalize_source @normalize_traces @epochs 250 @dir_data_train "../data/formatted/HEMEWS3D_S32_Z32_T320_fmax5_rot0_train" @dir_data_val "../data/formatted/HEMEWS3D_S32_Z32_T320_fmax5_rot0_val" @dir_logs "../logs/"
