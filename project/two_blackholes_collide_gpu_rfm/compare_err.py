import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

gpu_output="./out0d-conv_factor1.00.txt"
# cpu_output="../wavetoy_nosimd/out0d-conv_factor1.00.txt"

cpu_dict = {
    # "NOSIMD_NOPRE" : {'file' : "../wavetoy_nosimd_nopre/out0d-conv_factor1.00.txt",
    #                   'ls' : '-',
    #                   'alpha' : 1,
    #                   'lw' : 3},
    "NOSIMD_PRE"   : {'file' : "../two_blackholes_collide/out0d-conv_factor1.00.txt",
                      'ls' : '--',
                      'alpha' : 0.7,
                      'lw' : 3},
    # "NOSIMD_PRE_C"   : {'file' : "../two_blackholes_collide_clean/out0d-conv_factor1.00.txt",
    #                   'ls' : '--',
    #                   'alpha' : 0.7,
    #                   'lw' : 3},
    # "SIMD_NOPRE"   : {'file' : "../wavetoy_simd_nopre/out0d-conv_factor1.00.txt",
    #                   'ls' : '-',
    #                   'alpha' : 1,
    #                   'lw' : 3},
    # "SIMD_PRE"     : {'file' : "../wavetoy_simd_pre/out0d-conv_factor1.00.txt",
    #                   'ls' : '--',
    #                   'alpha' : 0.7,
    #                   'lw' : 3},
}

g_t, g_log10HL, g_log10sqrtM2L, g_cfL, g_alphaL, g_trKL =np.loadtxt(gpu_output, delimiter=' ',unpack=True)

axs=plt.gca()
axs.set_yscale('log')
axs.set_ylabel(r"Relative Diff GPU vs CPU @ P(0,0)")
axs.set_xlabel(r"t")
# axs.plot(g_t, np.fabs(1. - g_Unum/cs_Unum), label="CPU_SIMD_PRE")

for label, details in cpu_dict.items():
    f = details['file']
    ls = details['ls']
    alpha=details['alpha']
    lw = details['lw']
    
    c_t, c_log10HL, c_log10sqrtM2L, c_cfL, c_alphaL, c_trKL = np.loadtxt(f, delimiter=' ',unpack=True)
    
    axs.plot(g_t, 
             np.fabs(1. - g_log10HL/c_log10HL), 
             label=label,
             ls=ls,
             alpha=alpha,
             lw=lw)



plt.legend()
plt.show()