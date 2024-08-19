# %%
import csv

# %%
pathabs = "/home/tootle/lib/nrpy/project/cuda_profile.out"
f = open(pathabs, 'r')
csvreader = csv.reader(f, delimiter=',')

dict_keys = None
for l in csvreader:
    if l[0] == 'ID':
        dict_keys = l
        break

if dict_keys is None:
    raise RuntimeError("No key row found starting with ID")

data_dict = { k : [] for k in dict_keys }
for l in csvreader:
    # if len(l) == len(dict_keys):
    for i, k in enumerate(dict_keys):
        if len(l) <= len(dict_keys):
            if i < len(l):
                v = int(l[i]) if i == 0 else l[i]
                data_dict[k] += [v]
            else:
                data_dict[k] += ['']
f.close()

# id_data_dict = { k : [] for k in dict_keys }
# data_dict = {}
# for l in csvreader:
#     if len(l) == len(dict_keys):
#         for i, k in enumerate(dict_keys[1:]):
#             v = int(l[i]) if i == 0 else l[i]
#             id_data_dict[k] += [v]
#         data_dict.update({ int(l[0]) : id_data_dict })

scaling_dict = {
    'ID' : [],
    't'  : [],
    'DP' : [],
    'AI' : [],
}
f.close()
# table_started = False
# raw_read = False
# data_dict = dict()
# region = str()
# for l in f:
#     if "TABLE" in l and "Metric" in l:
#         table_started = True
#         raw_read = False
#         region = l.split(',')[1]
#         if not region in data_dict.keys():
#             data_dict[region] = dict()
#     elif "Raw" in l:
#         table_started = False
#         raw_read = True
#         region = l.split(',')[1]
#         if not region in data_dict.keys():
#             data_dict[region] = dict()
#     elif table_started:
#         data_dict_keys= ['DP', 'AI']
#         for k in data_dict_keys:
#             if k in l:
#                 data_dict[region][k] = float(l.split(',')[1])
#                 break
#     elif raw_read:
#         data_dict_keys= ['RDTSC', 'call count']
#         for k in data_dict_keys:
#             if k in l:
#                 data_dict[region][k] = float(l.split(',')[1])
#                 break
# f.close()

# # %%

# output_key_order = ['rhs_eval', 'residual', 'apply', 'rk_substep_1']
# header_elements = [ f"{Region} - {Key}" for Key in output_key_order for Region in list(data_dict.values())[0].keys()]
# header_str = ",".join(header_elements)
# #print(header_str)

# outstr = str()
# for k in output_key_order:
#     for K, I in data_dict.items():
#         if k in K:
#             CC = float(I['call count'])
#             t_ms = float(I['RDTSC']) * 1e3 / CC
#             outstr +=f"{t_ms:1.3f}, {I['DP']}, {I['AI']}, "
#             break
# print(outstr)
# # %%
