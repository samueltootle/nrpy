#!/bin/bash

workingdir="nrpyelliptic_conformally_flat_gpu-scaling-P128"
stdoutputfile="cuda_profile.out"

cd $workingdir
f1="SinhSymTP/numerical_grid_params_Nxx_dxx_xx__rfm__SinhSymTP.cu"
f2="params_struct_set_to_default.cu"

# cp ${f1} ${f1}.bak
# cp ${f2} ${f2}.bak
cp ${f1}.bak ${f1}
cp ${f2}.bak ${f2}
cd -

rm $stdoutputfile

pres=128
pphi=16
for RES in `seq 128 64 512`;do
  cd $workingdir

  phires=`printf %.0f $(echo "16 * (${RES} / 128)^2" | bc -l)`
  echo "Updating params to res $RES, $RES, $phires"
  sed -i -E "s/params->Nxx0 = ${pres}/params->Nxx0 = ${RES}/g"  $f1
  sed -i -E "s/params->Nxx1 = ${pres}/params->Nxx1 = ${RES}/g"  $f1
  sed -i -E "s/params->Nxx2 = ${pphi}/params->Nxx2 = ${phires}/g"  $f1

  sed -i -E "s/params->Nxx0 = ${pres}/params->Nxx0 = ${RES}/g"  $f2
  sed -i -E "s/params->Nxx1 = ${pres}/params->Nxx1 = ${RES}/g"  $f2
  sed -i -E "s/params->Nxx2 = ${pphi}/params->Nxx2 = ${phires}/g"  $f2

  echo "Update complete. Compiling"
  make -j;
  rm checkpoint-conv_factor1.00.dat
  ncu --csv \
    --kernel-id ::regex:".*rhs_eval.*|.*apply_bcs.*|.*rk_substep_1.*|.*compute_residual_all_points.*":1 \
    --set full \
    --target-processes all \
    --metrics sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,sm__sass_thread_inst_executed_op_dadd_pred_on.sum,sm__sass_thread_inst_executed_op_dfma_pred_on.sum,sm__sass_thread_inst_executed_op_dmul_pred_on.sum \
    nrpyelliptic_conformally_flat_gpu-scaling-P128 > ../$stdoutputfile

  cd -
  break
  #python3 cuda_parse.py >> cuda_scaling.log

  # Update previous resolutions to new ones
  pres=$RES
  pphi=$phires
done


