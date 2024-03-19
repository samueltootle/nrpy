#!/bin/bash

files=( "Spherical/initial_data_reader__convert_ADM_Cartesian_to_BSSN__rfm__Spherical.cu" "Spherical/constraints_eval__rfm__Spherical.cu" "Spherical/rhs_eval__rfm__Spherical.cu" "Spherical/Ricci_eval__rfm__Spherical.cu" )

for f in ${files[@]}; do
    sed -E -i 's/(const REAL)( .*Rational.*=)/constexpr REAL\2/' $f;
    sed -E -i 's/static/__device__ static/' $f;
done