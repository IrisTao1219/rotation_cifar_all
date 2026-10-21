"""Figures from the frozen confirmation results, including failure examples."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment import ROOT

CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


if __name__=='__main__':
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    report=json.loads((ROOT/'results/confirm_pool_max_large_vs_aug4_large.json').read_text())
    base=np.array(report['methods']['aug4_large']['accuracy_by_seed'])*100
    module=np.array(report['methods']['pool_max_large']['accuracy_by_seed'])*100
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7,4.7))
    colors=['#7b61a8','#168a85','#d18a28']
    for i,color in enumerate(colors):ax.plot([0,1],[base[i],module[i]],'o-',color=color,lw=1.7,label=f'Seed {i+1}',alpha=.85)
    for x,values in enumerate([base,module]):
        ax.scatter([x],[values.mean()],color='#172a3a',marker='D',s=65,zorder=4)
        ax.text(x,values.max()+.27,f'{values.mean():.2f}% mean',ha='center',weight='bold')
    ax.set_xticks([0,1],['Four-view training\n+ prediction averaging','Shared encoder\n+ feature maximum pooling'])
    ax.set_xlim(-.3,1.3);ax.set_ylim(71.8,77);ax.set_ylabel('Confirmation accuracy (%)')
    ax.set_title('CIFAR-10 quarter-turn classification',loc='left',weight='bold')
    ax.grid(axis='y',alpha=.2);ax.legend(loc='lower right',frameon=False)
    fig.text(.09,.02,'5,000 held-out source images; same parameters and 80 training epochs.',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,1));fig.savefig(out/'confirmation_results.png',dpi=180);plt.close(fig)
    recalls=[]
    for mode in ['aug4_large','pool_max_large']:
        matrices=[np.array(json.loads((ROOT/'runs'/f'{mode}_s{s}_e80'/'fixed_confirm.json').read_text())['confusion']) for s in [1,2,3]]
        matrix=np.sum(matrices,axis=0);recalls.append(np.diag(matrix)/matrix.sum(1)*100)
    fig,ax=plt.subplots(figsize=(8,4.8));positions=np.arange(10)
    ax.barh(positions-.17,recalls[0],height=.32,color='#8093a1',label='Prediction average')
    ax.barh(positions+.17,recalls[1],height=.32,color='#168a85',label='Feature maximum pool')
    ax.set_yticks(positions,CLASSES);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Per-class recall (%)')
    ax.set_title('All classes, mean over three seeds',loc='left',weight='bold');ax.legend(frameon=False,loc='lower right');ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(out/'per_class_results.png',dpi=180);plt.close(fig)
    b=np.load(ROOT/'runs/aug4_large_s1_e80/fixed_confirm_predictions.npz')
    m=np.load(ROOT/'runs/pool_max_large_s1_e80/fixed_confirm_predictions.npz')
    source=np.load(ROOT/'data/confirm.npz');assert np.array_equal(source['ids'],b['ids'])
    truth=b['labels'];bp=b['probabilities'].mean(0).argmax(1);mp=m['probabilities'].mean(0).argmax(1)
    improved=np.flatnonzero((bp!=truth)&(mp==truth))[:2];failed=np.flatnonzero((bp==truth)&(mp!=truth))[:2]
    selected=np.concatenate([improved,failed]);fig,axes=plt.subplots(4,5,figsize=(10,8))
    examples=[]
    for row,index in enumerate(selected):
        image=source['images'][index].transpose(1,2,0)
        for k in range(4):
            axes[row,k].imshow(np.rot90(image,k),interpolation='nearest');axes[row,k].axis('off')
            if row==0:axes[row,k].set_title(f'{k*90} degrees')
        ax=axes[row,4];ax.axis('off');kind='Corrected error' if row<2 else 'New error'
        ax.text(0,.85,kind,weight='bold',color='#14786f' if row<2 else '#b35049')
        ax.text(0,.58,f'Truth: {CLASSES[truth[index]]}\nBaseline TTA: {CLASSES[bp[index]]}\nFeature pool: {CLASSES[mp[index]]}',va='top',linespacing=1.6,fontsize=10)
        examples.append({'source_id':int(source['ids'][index]),'truth':CLASSES[truth[index]],'baseline':CLASSES[bp[index]],'module':CLASSES[mp[index]],'kind':kind})
    fig.suptitle('Both methods see all four directions',x=.05,ha='left',weight='bold')
    fig.text(.05,.015,'Seed 1: first two corrections and first two new errors in the frozen confirmation order.\nIllustrations are selected examples; the aggregate result uses all 5,000 images.',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.96));fig.savefig(out/'prediction_examples.png',dpi=180);plt.close(fig)
    (out/'example_selection.json').write_text(json.dumps(examples,indent=2))
    print(json.dumps({'figures':str(out),'examples':examples,'per_class_gain_pp':(recalls[1]-recalls[0]).tolist()},indent=2))
