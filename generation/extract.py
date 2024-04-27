s = open('./nohup.out','r').readlines()
best = 10000
besti = 0
print(len(s))
for i,line in enumerate(s):
    if line.startswith('valid:'):
        import re
        perp = re.findall(r'ppl: (\d)+\.(\d)+')[0]
        ind = perp.find(':')
        perp = perp[ind+1:]
        if best>float(perp):
            best = float(perp)
            besti = i
print('best perplexity',best)
print(besti)