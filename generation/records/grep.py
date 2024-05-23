import re
ls = open('./ablation_final_2.log').readlines()
out = []
for i,l in enumerate(ls):
    if l.find('ppl') != -1:
        out += [ls[i-1],l]
open('processed.log','w').writelines(out)