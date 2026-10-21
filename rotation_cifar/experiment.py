"""Quarter-turn robustness with augmentation and compute-matched controls."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent


class LeNet(nn.Module):
    def __init__(self,mode='plain'):
        super().__init__()
        self.mode=mode;self.invariant=mode.startswith('pool_') or mode in ['canon_soft','canon_hard','canon_aux_tta']
        first,second=(32,64) if mode.endswith('_large') else ((8,20) if mode=='aug4_wide' else (6,16))
        self.feature_dim=second*25
        self.encoder=nn.Sequential(nn.Conv2d(3,first,5),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(first,second,5),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten())
        self.classifier=nn.Sequential(nn.Linear(second*25,120),nn.ReLU(),nn.Linear(120,84),nn.ReLU(),nn.Linear(84,10))
        if mode.startswith('canon_'):self.orientation=nn.Sequential(nn.Linear(400,32),nn.ReLU(),nn.Linear(32,1))
        self.gate_logits=None;self.view_logits=None

    def forward(self,x):
        if self.invariant:
            feature=self.encoder(orbit(x)).reshape(4,len(x),self.feature_dim)
            if self.mode.startswith('pool_max'):return self.classifier(feature.max(0).values)
            if self.mode.startswith('pool_mean'):return self.classifier(feature.mean(0))
            self.gate_logits=self.orientation(feature)[:,:,0]
            if self.mode=='canon_aux_tta':
                self.view_logits=self.classifier(feature)
                return self.view_logits.softmax(2).mean(0).clamp_min(1e-8).log()
            if self.mode=='canon_hard':
                pooled=feature[self.gate_logits.argmax(0),torch.arange(len(x),device=x.device)]
            else:pooled=(feature*self.gate_logits.softmax(0)[:,:,None]).sum(0)
            return self.classifier(pooled)
        return self.classifier(self.encoder(x))


def data(split):
    a=np.load(ROOT/'data'/f'{split}.npz')
    x=torch.tensor(a['images'],device='cuda',dtype=torch.float32)/255
    mean=torch.tensor([.4914,.4822,.4465],device='cuda')[None,:,None,None]
    std=torch.tensor([.247,.2435,.2616],device='cuda')[None,:,None,None]
    return (x-mean)/std,torch.tensor(a['labels'],device='cuda',dtype=torch.long)


def crop_flip(x,generator):
    padded=F.pad(x,(4,4,4,4),mode='reflect').permute(0,2,3,1)
    offsets=torch.randint(9,(len(x),2),device='cuda',generator=generator);axis=torch.arange(32,device='cuda')
    x=padded[torch.arange(len(x),device='cuda')[:,None,None],offsets[:,0,None,None]+axis[None,:,None],offsets[:,1,None,None]+axis[None,None,:]].permute(0,3,1,2)
    flip=torch.rand(len(x),device='cuda',generator=generator)<.5
    return torch.where(flip[:,None,None,None],x.flip(-1),x)


def orbit(x):
    return torch.cat([x.rot90(k,(-2,-1)) for k in range(4)])


@torch.no_grad()
def evaluate(model,x,y):
    model.eval();blocks=[];orientation_correct=0
    for batch in x.split(256):
        if model.invariant:
            probs=model(batch).softmax(1)[None].expand(4,-1,-1)
            if model.gate_logits is not None:orientation_correct+=int((model.gate_logits.argmax(0)==0).sum())
        else:probs=model(orbit(batch)).softmax(1).reshape(4,len(batch),10)
        blocks.append(probs.cpu().numpy())
    p=np.concatenate(blocks,axis=1);truth=y.cpu().numpy();pred=p.argmax(2);tta=p.mean(0).argmax(1)
    correct=pred==truth[None];report={'angle_accuracy':correct.mean(1).tolist(),'mean_angle_accuracy':float(correct.mean()),'tta_accuracy':float((tta==truth).mean()),'all_angles_correct_rate':float(correct.all(0).mean()),'direction_disagreement_rate':float((pred!=pred[0]).any(0).mean()),'some_angle_correct_but_tta_wrong':int((correct.any(0)&(tta!=truth)).sum())}
    report['canonical_accuracy']=report['angle_accuracy'][0]
    if model.mode.startswith('canon_'):report['orientation_accuracy']=orientation_correct/len(y)
    model.train();return report,p


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['plain','aug','aug4','aug4_wide','aug4_large','pool_max','pool_max_large','pool_mean','pool_mean_large','canon_soft','canon_hard','canon_aux_tta'],required=True);p.add_argument('--seed',type=int,default=1);p.add_argument('--epochs',type=int,default=40);args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(args.seed);torch.cuda.set_per_process_memory_fraction(.2);torch.backends.cudnn.benchmark=True
    train,y=data('train');dev,dy=data('dev');model=LeNet(args.mode).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    sampler=torch.Generator(device='cuda').manual_seed(1234+args.seed);augment=torch.Generator(device='cuda').manual_seed(5000+args.seed)
    rotation_generator=torch.Generator(device='cuda').manual_seed(6000+args.seed)
    out=ROOT/'runs'/f'{args.mode}_s{args.seed}_e{args.epochs}';out.mkdir(parents=True,exist_ok=False)
    config=vars(args)|{'train_images':len(train),'parameters':sum(p.numel() for p in model.parameters()),'batch_original_images':256,'crop_flip':True,'selection':'canonical_accuracy' if args.mode=='plain' else 'tta_accuracy'}
    (out/'config.json').write_text(json.dumps(config,indent=2));print(json.dumps(config),flush=True)
    best=-1.;start=time.monotonic()
    with (out/'progress.jsonl').open('w') as log:
        for epoch in range(1,args.epochs+1):
            lr=.00001+(.001-.00001)*(1+math.cos(math.pi*(epoch-1)/args.epochs))/2
            for group in optimizer.param_groups:group['lr']=lr
            order=torch.randperm(len(train),device='cuda',generator=sampler);total=0.
            for ids in order.split(256):
                x=crop_flip(train[ids],augment);target=y[ids]
                if args.mode=='aug':
                    k=torch.randint(4,(len(x),),device='cuda',generator=rotation_generator)
                    x=torch.stack([x.rot90(i,(-2,-1)) for i in range(4)])[k,torch.arange(len(x),device='cuda')]
                elif args.mode.startswith('aug4'):x=orbit(x);target=target.repeat(4)
                loss=F.cross_entropy(model(x),target)
                if args.mode.startswith('canon_'):
                    if args.mode=='canon_aux_tta':loss=F.cross_entropy(model.view_logits.reshape(-1,10),target.repeat(4))
                    loss=loss+.2*F.cross_entropy(model.gate_logits.T,torch.zeros(len(x),device='cuda',dtype=torch.long))
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();total+=float(loss)*len(ids)
            report,probability=evaluate(model,dev,dy);score=report[config['selection']]
            row={'epoch':epoch,'loss':total/len(train),'dev':report,'seconds':time.monotonic()-start}
            if score>best:
                best=score;torch.save({'model':model.state_dict(),'config':config,'epoch':epoch,'dev':report},out/'best.pt');np.savez_compressed(out/'best_dev_predictions.npz',probabilities=probability,labels=dy.cpu().numpy())
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    (out/'complete.json').write_text(json.dumps({'best_score':best,'last':report,'seconds':time.monotonic()-start},indent=2))
