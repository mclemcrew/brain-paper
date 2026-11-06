"""Unified Holm family for Table 2: all adapter-vs-base rows per metric (anchors and base_sec excluded),
pooled across eval/tier1 and eval/tier3 comparisons.csv; B6k (present in both tiers, identical raw p) counted once.
Usage: python3 holm_unified.py <repo_root> [--table-only]"""
import csv, sys
root = sys.argv[1]
ADAPTERS = ['A6k','A6k_sec','B2k','B4k','B6k','B6k_sec','C6k','D6k','E6k','E6k_nosec']
METRICS = [('region_sim_clap','track'),('region_sim_mert','track'),('region_sim_mert_centred','track'),('clap_text','clip')]
rows = {}
for t in ['tier1','tier3']:
    for r in csv.DictReader(open(f'{root}/eval/{t}/comparisons.csv')):
        if r['cond_b']=='base' and r['cond_a'] in ADAPTERS and (r['metric'],r['level']) in METRICS:
            key=(r['metric'],r['cond_a'])
            if key in rows:
                assert abs(float(rows[key]['p_two_sided'])-float(r['p_two_sided']))<1e-12
                continue
            r['tier']=t; rows[key]=r
def holm(ps):
    m=len(ps); order=sorted(range(m),key=lambda i:ps[i]); adj=[0]*m; run=0
    for k,i in enumerate(order):
        run=max(run,min(1.0,(m-k)*ps[i])); adj[i]=run
    return adj
out={}
for met,_ in METRICS:
    conds=[c for c in ADAPTERS if (met,c) in rows]
    ps=[float(rows[(met,c)]['p_two_sided']) for c in conds]
    for c,a in zip(conds,holm(ps)):
        r=rows[(met,c)]; out[(met,c)]=a
        print(f"{met:24s} {c:10s} m={len(conds)} n={r['n']} median_diff={float(r['median_diff']):+.4f} ci=[{float(r['ci_low']):+.4f},{float(r['ci_high']):+.4f}] p_raw={float(r['p_two_sided']):.3g} p_holm_unified={a:.3g} (csv holm {float(r['p_holm_two_sided']):.3g}, m={r['holm_m']}, {r['tier']})")
    print()
