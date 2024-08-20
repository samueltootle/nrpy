# %%
import csv, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--print_header", action="store_true", default=False, help="Print CSV style header information")
parser.add_argument("--file",
                    "-f",
                    type=str,
                    help="CUDA NCU file to parse",
                    default="/home/tootle/lib/nrpy/project/cuda_profile.out")
args = parser.parse_args()

# %%
pathabs = args.file
f = open(pathabs, 'r')
csvreader = csv.reader(f, delimiter=',')

# %%
dict_keys = None
for l in csvreader:
    if l[0] == 'ID':
        dict_keys = l
        break

if dict_keys is None:
    raise RuntimeError("No key row found starting with ID")

dict_key_map = { key : i for i, key in enumerate(dict_keys) }

# %%
id_data_dict = { k : [] for k in dict_keys }
data_dict = {}

for iCSV, lCSV in enumerate(csvreader):
    intermediate_dict = {}
    idx = int(lCSV[dict_key_map['ID']])
    kernel = lCSV[dict_key_map['Kernel Name']]
    metric = lCSV[dict_key_map['Metric Name']]

    try:
        metric_val = float(lCSV[dict_key_map['Metric Value']].replace(",", ""))
    except Exception as exp:
        metric_val = lCSV[dict_key_map['Metric Value']]

    # metric_val = lCSV[dict_key_map['Metric Value']]
    metric_unit = lCSV[dict_key_map['Metric Unit']]
    if idx in data_dict:
        data_dict[idx][kernel].update({metric : { 'value' :  metric_val, 'unit' : metric_unit}})
    else:
        data_dict.update({
            idx : {
                kernel : {
                    metric : { 'value' : metric_val, 'unit' : metric_unit}
                }
            }
        })
f.close()

# %%
kernel_names = {
    'H' : 'residual',
    'RHS' : 'rhs',
    'RK' : 'rk_substep',
    'BC' : 'radiation'
}

scaling_dict = {
    'ID' : 0,
    't'  : 0,
    'DFLOPS' : 0,
    'FFLOPS' : 0,
    'DP' : 0,
    'FP' : 0,
    'DAI' : 0,
    'FAI' : 0,
}

# %%
all_scaling_dict = { k : scaling_dict.copy() for k in kernel_names }

for id in data_dict:
    found = False
    for Key, filter in kernel_names.items():
        kernel = list(data_dict[id].keys())[0]
        if filter in kernel:
            # print(kernel, filter)
            found = True
            break
    if found is False:
        continue

    # Get duration in seconds
    t = data_dict[id][kernel]['Duration']['value']
    t_unit = data_dict[id][kernel]['Duration']['unit']
    if t_unit == 'nsecond':
        t = t * 1e-9
    else:
        raise RuntimeError(f"Time conversion not available for {t_unit}")
    # End duration

    # Compute FLOPS
    dp_FLOPS = 0
    fp_FLOPS = 0
    for data_key in data_dict[id][kernel]:
        if 'sm__sass' in data_key:
            if 'dadd' in data_key or 'dmul' in data_key:
                dp_FLOPS += data_dict[id][kernel][data_key]['value']
            elif 'fadd' in data_key or 'fmul' in data_key:
                fp_FLOPS += data_dict[id][kernel][data_key]['value']
            elif 'dfma' in data_key:
                dp_FLOPS += data_dict[id][kernel][data_key]['value'] * 2
            elif 'ffma' in data_key:
                fp_FLOPS += data_dict[id][kernel][data_key]['value'] * 2
    # End compute FLOPS

    # Compute Arithmetic intensity
    Mem_throughput = data_dict[id][kernel]['Memory Throughput']['value']
    Total_memory = Mem_throughput * t
    DAI = dp_FLOPS / Total_memory
    FAI = fp_FLOPS / Total_memory
    # End compute Arithmetic intensity

    all_scaling_dict[Key]['ID'] = id
    all_scaling_dict[Key]['t'] = t
    all_scaling_dict[Key]['DFLOPS'] = dp_FLOPS
    all_scaling_dict[Key]['FFLOPS'] = fp_FLOPS
    all_scaling_dict[Key]['DP'] = dp_FLOPS / t * 1e-9
    all_scaling_dict[Key]['FP'] = fp_FLOPS / t * 1e-9
    all_scaling_dict[Key]['DAI'] = DAI
    all_scaling_dict[Key]['FAI'] = FAI

# %%

output_key_order = ['RHS', 'H', 'BC', 'RK']
output_subkeys = ['Duration (ms)', 'Float GFLOPS/s', 'Float AI (FLOPS/byte)', 'Double GFLOPS/s', 'Double AI (FLOPS/byte)']
header_elements = [f"{kernal_abrv} - {perf}" for kernal_abrv in output_key_order for perf in output_subkeys ]
header_str = ",".join(header_elements)
if args.print_header:
    print(header_str)

outstr = str()
for k in output_key_order:
    for K, I in all_scaling_dict.items():
        if k in K:
            t_ms = float(I['t']) * 1e3 # milliseconds
            outstr +=f"{t_ms:1.3f}, {I['FP']}, {I['FAI']}, {I['DP']}, {I['DAI']}, "
            break
print(outstr)
# # %%

# %%
