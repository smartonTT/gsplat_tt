import csv,sys,collections
for fn in sys.argv[1:]:
    s=collections.defaultdict(float)
    with open(fn) as f:
        next(f);next(f)
        for r in csv.reader(f):
            if len(r)>10 and r[10].startswith('fz_mv_'): s[(r[3],r[10])]+=float(r[6])
    print(fn)
    for risc in ('BRISC','NCRISC'):
        n=s[(risc,'fz_mv_recs')]
        print(' ',risc,'recs',int(n),' '.join(f"{k[6:]}={s[(risc,k)]/n:.1f}" for k in ('fz_mv_sort','fz_mv_perm','fz_mv_rd','fz_mv_wr','fz_mv_big') if n))
