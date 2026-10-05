# Straight-line blocks of a TRISC1 objdump with >=20 SFPU ops (the unrolled
# blend dispatch bodies), classified: rt_push = runtime-built Tensix word
# pushes (every sw in these blocks), k_loadi = constant SFPLOADI, risc = other scalar insns.
# Usage: insn_mix.py trisc1.dis

import re,sys,collections
L=[]
for ln in open(sys.argv[1] if len(sys.argv)>1 else 'trisc1.dis'):
    p=ln.rstrip('\n').split('\t')
    if len(p)>=3 and re.match(r'\s*[0-9a-f]+:',p[0]):
        L.append((int(p[0].strip()[:-1],16),p[2].split()[0] if p[2].strip() else '', '\t'.join(p[2:])))
br={'j','jal','ret','beq','bne','beqz','bnez','bltu','bgeu','blt','bge','jalr'}
# segments = straight-line blocks between control transfers
segs=[];cur=[]
for a,m,t in L:
    cur.append((a,m,t))
    if m in br: segs.append(cur);cur=[]
def cls(seg):
    c=collections.Counter()
    for a,m,t in seg:
        if m=='sw': c['rt_push']+=1
        elif m=='sfploadi': c['k_loadi']+=1
        elif m=='sfpnop': c['sfpnop']+=1
        elif m.startswith('sfp'): c['sfp_other']+=1
        elif m.startswith('tt'): c['tt']+=1
        else: c['risc']+=1
    return c
out=[]
for s in segs:
    c=cls(s)
    if c['sfp_other']>=20: out.append((hex(s[0][0]),len(s),dict(c)))
for o in out: print(o)
