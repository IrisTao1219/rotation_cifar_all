"""Apply each seed's simple aggregation choice fixed on development data."""
import json

import numpy as np

from experiment import ROOT


if __name__=='__main__':
    lock=json.loads((ROOT/'confirmation_lock.json').read_text());rows=[];differences=[]
    for seed in lock['seeds']:
        base=ROOT/'runs'/f'aug4_large_s{seed}_e80';module=ROOT/'runs'/f'pool_max_large_s{seed}_e80'
        selection=json.loads((base/'aggregation_selection.json').read_text())['selected']
        a=np.load(base/'fixed_confirm_predictions.npz');b=np.load(module/'fixed_confirm_predictions.npz');p=a['probabilities'];y=a['labels']
        assert np.array_equal(y,b['labels']) and np.array_equal(a['ids'],b['ids'])
        confidence=p.max(2);entropy=-(p*np.log(p.clip(1e-8))).sum(2)
        variants={'mean':p.mean(0),'geometric':np.log(p.clip(1e-8)).mean(0),'confidence':(p*confidence[:,:,None]).sum(0),'inverse_entropy':(p/(entropy[:,:,None]+.01)).sum(0),'max_confidence':p[confidence.argmax(0),np.arange(len(y))]}
        chosen=variants[selection].argmax(1)==y;proposed=b['probabilities'].mean(0).argmax(1)==y
        differences.append(proposed.astype(float)-chosen)
        rows.append({'seed':seed,'aggregation_selected_on_dev':selection,'baseline_accuracy':float(chosen.mean()),'module_accuracy':float(proposed.mean()),'difference':float(proposed.mean()-chosen.mean())})
    d=np.stack(differences);rng=np.random.default_rng(20261011);samples=[]
    for _ in range(5000):
        seeds=rng.integers(3,size=3);images=rng.integers(d.shape[1],size=d.shape[1]);samples.append(float(d[seeds][:,images].mean()))
    report={'selection_split':'dev','evaluation_split':'confirm','rows':rows,'mean_difference':float(d.mean()),'paired_image_and_seed_bootstrap_95_interval':np.percentile(samples,[2.5,97.5]).tolist()}
    (ROOT/'results/strong_control_confirmation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
