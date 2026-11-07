"""Matched-seed descriptive comparison; all registered attempts are retained."""
import json,statistics,sys
from pathlib import Path


def compare(experiment, baseline):
    report=json.loads((experiment/'report.json').read_text())
    base=json.loads(baseline.read_text())
    assert report['baseline_complete'] and report['source_integrity'] and report['official_files_unchanged']
    assert len(report['summary_rows'])==18
    controls={(str(r['world']),r['environment_seed']):r for r in base['summary_rows']}
    pairs=[]
    for row in report['summary_rows']:
        control=controls[str(row['world']),row['environment_seed']]
        fv=0. if row['total'] is None else row['total']
        bv=0. if control['total'] is None else control['total']
        pairs.append({'world':str(row['world']),'round':row['repeat'],'seed':row['environment_seed'],
                      'baseline':bv,'filter':fv,'delta':fv-bv,
                      'baseline_success':control['success'],'filter_success':row['success']})
    assert len({(r['world'],r['seed']) for r in pairs})==18
    worlds={}
    for w in range(1,7):
        rows=[r for r in pairs if r['world']==str(w)];assert len(rows)==3
        worlds[str(w)]={'baseline_mean':statistics.mean(r['baseline'] for r in rows),
                       'filter_mean':statistics.mean(r['filter'] for r in rows),
                       'filter_sample_std':statistics.stdev(r['filter'] for r in rows),
                       'mean_delta':statistics.mean(r['delta'] for r in rows),
                       'baseline_success':sum(r['baseline_success'] for r in rows),
                       'filter_success':sum(r['filter_success'] for r in rows),
                       'scores':[r['filter'] for r in rows],'paired_deltas':[r['delta'] for r in rows]}
    result={'experiment':experiment.name,'pairs':pairs,'worlds':worlds,
            'matched_baseline_mean':statistics.mean(r['baseline'] for r in pairs),
            'filter_mean':statistics.mean(r['filter'] for r in pairs),
            'mean_delta':statistics.mean(r['delta'] for r in pairs),
            'baseline_success':sum(r['baseline_success'] for r in pairs),
            'filter_success':sum(r['filter_success'] for r in pairs),
            'full_five_repeat_baseline_mean':base['overall_mean'],
            'scope':'Descriptive three-repeat comparison. Model seed unset; matched environment seeds do not imply identical observation noise after different actions. E13 excluded.'}
    (experiment/'baseline-comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result

if __name__=='__main__':
    r=compare(Path(sys.argv[1]),Path(sys.argv[2]));print(json.dumps({k:v for k,v in r.items() if k not in ['pairs','worlds']},ensure_ascii=False))
