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
    "NOSIMD_PRE"   : {'folder' : "../two_blackholes_collide/O0/",
                      'ls' : '--',
                      'alpha' : 0.7,
                      'lw' : 3},
    #"NOSIMD_PRE_C"   : {'folder' : "../two_blackholes_collide_nosimd_rfm_clean",
    #                  'ls' : '--',
    #                  'alpha' : 0.7,
    #                  'lw' : 3},
    # "SIMD_NOPRE"   : {'file' : "../wavetoy_simd_nopre/out0d-conv_factor1.00.txt",
    #                   'ls' : '-',
    #                   'alpha' : 1,
    #                   'lw' : 3},
    #  "GPU_PERT"     : {'folder' : "./output-pert",
    #                    'ls' : '--',
    #                    'alpha' : 0.7,
    #                    'lw' : 3},
}
CPU = True
def plot(direction,f, directory=None):
    def get_time():
        start = f.find("-t")+2
        stop  = f.find(".txt")
        return f[start:stop]
    t = get_time()
    direction_file = f if directory == None else f.replace(directory,"")
    
    if CPU:
        gpu_output=f"../two_blackholes_collide/{direction_file}"
    else:
        gpu_output=f"./{f}"
    g_xx, g_log10HL, g_log10sqrtM2L, g_cfL, g_alphaL, g_trKL =np.loadtxt(gpu_output, delimiter=' ',unpack=True)

    fig = plt.figure(tight_layout=True)
    axs = fig.subplot_mosaic([['Top', 'Top'],['BottomLeft', 'BottomRight']])
    
    for ax in axs:
        # axs[ax].set_ylabel(r"$\log_{10} {\rm HL}$")
        axs[ax].set_ylabel(r"$\alpha$")
        axs[ax].set_xlabel(rf"{direction}")
        # axs[ax].set_ylim(-7, -2)
    axs['Top'].set_title(rf"t = {t}")
    if CPU:
        axs['Top'].set_ylabel(rf"Rel. Diff CPUO0 vs CPUO2 (alphaL)")
    else:
        axs['Top'].set_ylabel(rf"Rel. Diff GPU vs CPUO0 (alphaL)")
    axs['Top'].set_yscale('log')
    axs['Top'].set_ylim(1e-15, 1e-5)
    # axs.plot(g_t, np.fabs(1. - g_Unum/cs_Unum), label="CPU_SIMD_PRE")

    for label, details in cpu_dict.items():
        f = f"{details['folder']}/{direction_file}"
        ls = details['ls']
        alpha=details['alpha']
        lw = details['lw']
        
        print(gpu_output, f)
        c_xx, c_log10HL, c_log10sqrtM2L, c_cfL, c_alphaL, c_trKL = np.loadtxt(f, delimiter=' ',unpack=True)
        axs['BottomLeft'].scatter(g_xx, 
                # np.fabs(1. - g_log10HL/c_log10HL), 
                g_alphaL,
                # np.fabs(1. - g_log10sqrtM2L/c_log10sqrtM2L), 
                # label=f"{label}_{direction}",
                ls=ls,
                alpha=alpha,
                lw=lw)
        axs['BottomRight'].scatter(c_xx, 
                # np.fabs(1. - g_log10HL/c_log10HL), 
                c_alphaL,
                # np.fabs(1. - g_log10sqrtM2L/c_log10sqrtM2L), 
                # label=f"{label}_{direction}",
                ls=ls,
                alpha=alpha,
                lw=lw)  
        axs['Top'].scatter(g_xx, 
                np.fabs(1. - g_alphaL/c_alphaL), 
                # np.fabs(1. - g_log10sqrtM2L/c_log10sqrtM2L), 
                # label=f"{label}_{direction}",
                ls=ls,
                alpha=alpha,
                lw=lw)

for direction in ['y','z']:
    for i, f in enumerate(sorted(glob.glob(f"{out1d_files[direction]}*.*"))):
        plt.close('all')
        plot(direction, f, directory="")
        fnout = f"out-CPU/{direction}-{i:02d}.png" if CPU else f"out-O0/{direction}-{i:02d}.png"
        # plt.legend()
        # plt.ylim(-7, -2)
        plt.savefig(fnout)
    # break
# plot('y')
# plot('z')

