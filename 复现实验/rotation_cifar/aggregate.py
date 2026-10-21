"""Accuracy uncertainty using paired source images and paired training seeds."""
import argparse
import json

import numpy as np

from experiment import ROOT


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','confirm'],required=True);p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--seeds',type=int,nargs='+',default=[1,2,3]);p.add_argument('--modes',nargs='+',required=True);p.add_argument('--reference',required=True);p.add_argument('--candidate',default='pool_max');args=p.parse_args()
    methods={};correct={};ids=None;truth=None
    for mode in args.modes:
        scores=[];blocks=[];macro=[]
        for seed in args.seeds:
            run=ROOT/'runs'/f'{mode}_s{seed}_e{args.epochs}';a=np.load(run/f'fixed_{args.split}_predictions.npz')
            if ids is None:ids=a['ids'];truth=a['labels']
            assert np.array_equal(ids,a['ids']) and np.array_equal(truth,a['labels'])
            hits=a['probabilities'].mean(0).argmax(1)==a['labels'];blocks.append(hits);scores.append(float(hits.mean()))
            report=json.loads((run/f'fixed_{args.split}.json').read_text());macro.append(report['macro_f1'])
        correct[mode]=np.stack(blocks)
        methods[mode]={'accuracy_by_seed':scores,'mean_accuracy':float(np.mean(scores)),'sd_accuracy':float(np.std(scores,ddof=1)) if len(scores)>1 else None,'mean_macro_f1':float(np.mean(macro))}
    rng=np.random.default_rng(20261011);deltas=correct[args.candidate].astype(float)-correct[args.reference];samples=[]
    for _ in range(5000):
        seeds=rng.integers(len(args.seeds),size=len(args.seeds));images=rng.integers(len(ids),size=len(ids))
        samples.append(float(deltas[seeds][:,images].mean()))
    result={'split':args.split,'epochs':args.epochs,'seeds':args.seeds,'original_image_count':len(ids),'methods':methods,'comparison':{'candidate':args.candidate,'reference':args.reference,'difference_by_seed':deltas.mean(1).tolist(),'mean_difference':float(deltas.mean()),'paired_image_and_seed_bootstrap_95_interval':np.percentile(samples,[2.5,97.5]).tolist(),'bootstrap_draws':5000,'bootstrap_units':'original images and paired training seeds; the four rotations are not independent samples'}}
    out=ROOT/'results';out.mkdir(exist_ok=True);(out/f'{args.split}_{args.candidate}_vs_{args.reference}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
