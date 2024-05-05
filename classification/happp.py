l=open('./files.txt').readlines()
l=[int(x.strip()) for x in l]
print(sum(l)/len(l))
print(sum([c>1200 for c in l])/len(l))