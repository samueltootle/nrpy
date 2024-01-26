import numpy as np, glob
import matplotlib as mpl
import matplotlib.pyplot as plt

out1d_files= {'y' : "out1d-y-conv_factor1.00-",
              'z' : "out1d-z-conv_factor1.00-",
              }


# cpu_output="../wavetoy_nosimd/out0d-conv_factor1.00.txt"

cpu_dict = {
    # "NOSIMD_NOPRE" : {'file' : "../wavetoy_nosimd_nopre/out0d-conv_factor1.00.txt",
    #                   'ls' : '-',
    #                   'alpha' : 1,
    #                   'lw' : 3},
    # "NOSIMD_PRE"   : {'folder' : "../two_blackholes_collide/",
    #                   'ls' : '--',
    #                   'alpha' : 0.7,
    #                   'lw' : 3},
    #"NOSIMD_PRE_C"   : {'folder' : "../two_blackholes_collide_nosimd_rfm_clean",
    #                  'ls' : '--',
    #                  'alpha' : 0.7,
    #                  'lw' : 3},
    # "SIMD_NOPRE"   : {'file' : "../wavetoy_simd_nopre/out0d-conv_factor1.00.txt",
    #                   'ls' : '-',
    #                   'alpha' : 1,
    #                   'lw' : 3},
     "SIMD_PRE"     : {'folder' : "../two_blackholes_collide_simd_rfm_clean",
                       'ls' : '--',
                       'alpha' : 0.7,
                       'lw' : 3},
}

def plot(direction,f):
    def get_time():
        start = f.find("-t")+2
        stop  = f.find(".txt")
        return f[start:stop]
    t = get_time()
    direction_file = f
    gpu_output=f"./{direction_file}"
    g_xx, g_log10HL, g_log10sqrtM2L, g_cfL, g_alphaL, g_trKL =np.loadtxt(gpu_output, delimiter=' ',unpack=True)

    fig = plt.figure(tight_layout=True)
    axs = fig.subplot_mosaic([['Top', 'Top'],['BottomLeft', 'BottomRight']])
    
    for ax in axs:
        axs[ax].set_ylabel(r"$\log_{10} {\rm HL}$")
        axs[ax].set_xlabel(r"t")
        axs[ax].set_ylim(-7, -2)
    axs['Top'].set_title(rf"t = {t}")
    axs['Top'].set_ylabel(rf"Rel. Diff GPU vs CPU ({direction})")
    axs['Top'].set_yscale('log')
    axs['Top'].set_ylim(1e-12, 1e-5)
    # axs.plot(g_t, np.fabs(1. - g_Unum/cs_Unum), label="CPU_SIMD_PRE")

    for label, details in cpu_dict.items():
        f = f"{details['folder']}/{direction_file}"
        ls = details['ls']
        alpha=details['alpha']
        lw = details['lw']
        
        # print(f)
        c_xx, c_log10HL, c_log10sqrtM2L, c_cfL, c_alphaL, c_trKL = np.loadtxt(f, delimiter=' ',unpack=True)
        axs['BottomLeft'].scatter(g_xx, 
                # np.fabs(1. - g_log10HL/c_log10HL), 
                g_log10HL,
                # np.fabs(1. - g_log10sqrtM2L/c_log10sqrtM2L), 
                # label=f"{label}_{direction}",
                ls=ls,
                alpha=alpha,
                lw=lw)
        axs['BottomRight'].scatter(c_xx, 
                # np.fabs(1. - g_log10HL/c_log10HL), 
                c_log10HL,
                # np.fabs(1. - g_log10sqrtM2L/c_log10sqrtM2L), 
                # label=f"{label}_{direction}",
                ls=ls,
                alpha=alpha,
                lw=lw)  
        axs['Top'].plot(g_xx, 
                np.fabs(1. - g_log10HL/c_log10HL), 
                # np.fabs(1. - g_log10sqrtM2L/c_log10sqrtM2L), 
                # label=f"{label}_{direction}",
                ls=ls,
                alpha=alpha,
                lw=lw)

for direction in ['y','z']:
    for i, f in enumerate(sorted(glob.glob(f"{out1d_files[direction]}*.*"))):
        plt.close('all')
        plot(direction, f)
        # plt.legend()
        # plt.ylim(-7, -2)
        plt.savefig(f"out/{direction}-{i:02d}.png")
    # break
# plot('y')
# plot('z')

