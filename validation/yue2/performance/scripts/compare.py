"""Compare both order-balanced pairs without averaging request-level rates."""
import json
import math
from pathlib import Path

root=Path(__file__).resolve().parents[1]
analysis=json.loads((root/'artifacts/analysis.json').read_text())
rows={r['label']:r for r in analysis['results']}
assert list(rows)==['upstream-01','candidate-01','candidate-02','upstream-02']

def values(row):
    return dict(throughput=row['throughput'],request_throughput=row['request_throughput'],
                e2el_p50=row['e2el']['p50'],e2el_p95=row['e2el']['p95'],
                rtf_p50=row['rtf']['p50'],rtf_p95=row['rtf']['p95'],
                total_content_tps=row['total_content_tps'],semantic_tps=row['semantic_tps'],
                peak_gib=row['peak_gib'],allocated_gib=row['allocated_gib'],reserved_gib=row['reserved_gib'])
pairs=[]
for n in ['01','02']:
    a,b=rows[f'upstream-{n}'],rows[f'candidate-{n}']
    av,bv=values(a),values(b)
    pairs.append(dict(pair=n,order='AB' if n=='01' else 'BA',upstream=av,candidate=bv,
                      change_pct={k:100*(bv[k]/av[k]-1) for k in av},
                      peak_delta_gib=bv['peak_gib']-av['peak_gib'],
                      audio_duration_change_pct=100*(b['total_audio_s']/a['total_audio_s']-1)))
aggregate={}
for arm in ['upstream','candidate']:
    selected=[r for r in rows.values() if r['arm']==arm]
    wall=sum(r['elapsed_s'] for r in selected)
    aggregate[arm]=dict(elapsed_s=wall,throughput=sum(r['total_audio_s'] for r in selected)/wall,
                        total_content_tps=sum(r['semantic_tokens']+r['abc_content_tokens'] for r in selected)/wall,
                        max_peak_gib=max(r['peak_gib'] for r in selected))
result=dict(complete=True,pairs=pairs,aggregate=aggregate,
            pair_geometric_mean_change_pct={k:100*(math.sqrt((1+pairs[0]['change_pct'][k]/100)*(1+pairs[1]['change_pct'][k]/100))-1) for k in pairs[0]['change_pct']},
            limitation='One ABBA cycle, two matched pairs; descriptive results without statistical significance or quality claims.')
(root/'artifacts/comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
