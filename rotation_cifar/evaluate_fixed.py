"""Evaluate frozen development-selected models on untouched original images."""
import argparse
import json

import numpy as np
import torch

from experiment import ROOT,LeNet,data,evaluate


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--split',choices=['dev','confirm'],default='dev');args=p.parse_args()
    if args.split=='confirm':
        lock=json.loads((ROOT/'confirmation_lock.json').read_text())
        assert args.run in lock['runs']
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.2)
    run=ROOT/'runs'/args.run;checkpoint=torch.load(run/'best.pt',weights_only=True)
    model=LeNet(checkpoint['config']['mode']).cuda();model.load_state_dict(checkpoint['model'])
    x,y=data(args.split);report,probabilities=evaluate(model,x,y)
    labels=y.cpu().numpy();pred=probabilities.mean(0).argmax(1)
    confusion=np.bincount(labels*10+pred,minlength=100).reshape(10,10)
    tp=np.diag(confusion);class_f1=2*tp/np.maximum(1,confusion.sum(0)+confusion.sum(1))
    # Verify rather than merely assuming the invariant inference shortcut.
    model.eval();deviation=None
    if model.invariant:
        with torch.no_grad():
            source=model(x[:256]);deviation=max(float((source-model(x[:256].rot90(k,(-2,-1)))).abs().max()) for k in range(4))
        assert deviation<1e-4,deviation
    result={'run':args.run,'split':args.split,'epoch_selected_on_dev':checkpoint['epoch'],'parameters':checkpoint['config']['parameters'],'metrics':report,'macro_f1':float(class_f1.mean()),'per_class_f1':class_f1.tolist(),'confusion':confusion.tolist(),'max_rotation_logit_deviation_first256':deviation,'source_image_count':len(y)}
    (run/f'fixed_{args.split}.json').write_text(json.dumps(result,indent=2))
    ids=np.load(ROOT/'data'/f'{args.split}.npz')['ids']
    np.savez_compressed(run/f'fixed_{args.split}_predictions.npz',probabilities=probabilities,labels=labels,ids=ids)
    print(json.dumps(result),flush=True)
