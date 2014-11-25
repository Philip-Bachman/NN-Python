import os as os
import sys as sys
import numpy as np
import numpy.random as npr

def print_res(res):
    print("err: {0:.4f}".format(res['err']))
    for h_param in ['learn_rate', 'lam_cat', 'lam_pea', 'lam_ent', 'lam_l2w']:
        print("    {0:s}: {1:.4f}".format(h_param, res[h_param]))
    return 1

def parse_file(f_name):
    f_lines = [l for l in open(f_name).readlines()]
    f_dict = {}
    for i in [1, 2, 3, 4, 5]:
        f_dict[f_lines[i].split()[0].strip(':')] = float(f_lines[i].split()[1])
    e_lines = [l for l in f_lines if ('va_err:' in l)]
    e_vals = [float(l.split()[-1]) for l in e_lines]
    mean_err = sum(e_vals[-10:]) / len(e_vals[-10:])
    f_dict['err'] = mean_err
    return f_dict

comp_func = lambda x, y: 1 if (x['err'] > y['err']) else -1

if __name__=="__main__":
	if (len(sys.argv) < 2):
		print("FILE TAG REQUIRED!")
		assert(False)
	res_dicts = [parse_file(f) for f in os.listdir(os.getcwd()) if (sys.argv[1] in f)]
	res_dicts.sort(cmp=comp_func)
	print("**RESULTS**")
	for rd in res_dicts:
		print("========================================")
		print_res(rd)
