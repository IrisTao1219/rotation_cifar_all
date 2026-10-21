"""Decode the checksum-pinned training-only mirror without accessing test."""
import hashlib
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
import numpy as np
from PIL import Image
import pyarrow.parquet as pq


if __name__=='__main__':
    path=ROOT/'data/train.parquet';checksum='8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==checksum
    table=pq.read_table(path);labels=np.array(table['label'].to_pylist())
    images=np.stack([np.asarray(Image.open(io.BytesIO(r['bytes'])).convert('RGB')).transpose(2,0,1) for r in table['img'].to_pylist()])
    assert images.shape==(50000,3,32,32)
    groups={}
    for i,image in enumerate(images):
        key=min(hashlib.sha256(np.rot90(image,k,axes=(-2,-1)).tobytes()).hexdigest() for k in range(4))
        groups.setdefault(key,[]).append(i)
    unique=[ids[0] for ids in groups.values() if len(set(labels[ids].tolist()))==1]
    rng=np.random.default_rng(20261011);parts={'train':[],'dev':[],'confirm':[]}
    for c in range(10):
        ids=np.array([i for i in unique if labels[i]==c]);rng.shuffle(ids)
        parts['dev'].extend(ids[:500]);parts['confirm'].extend(ids[500:1000]);parts['train'].extend(ids[1000:])
    for split,ids in parts.items():
        ids=np.array(ids);rng.shuffle(ids);parts[split]=ids.tolist()
        np.savez_compressed(ROOT/'data'/f'{split}.npz',images=images[ids],labels=labels[ids],ids=ids)
    manifest={'source':'https://www.cs.toronto.edu/~kriz/cifar.html','mirror':'uoft-cs/cifar10','revision':'0b2714987fa478483af9968de7c934580d0bb9a2','train_parquet_sha256':checksum,'splits':parts,'counts':{k:len(v) for k,v in parts.items()},'duplicate_or_conflicting_images_excluded':50000-len(unique),'official_test_decoded_or_evaluated':False,'mirror_test_downloaded':False,'split_before_rotation':True}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2));print({k:v for k,v in manifest.items() if k!='splits'},flush=True)
