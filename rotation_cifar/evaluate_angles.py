"""Frozen-checkpoint evaluation on unseen rotations; no training or model selection."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from experiment import ROOT, LeNet, orbit


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rotate(images, angle):
    """Positive angles are counterclockwise; black fill, fixed 32x32 canvas."""
    angle = angle % 360
    if angle % 90 == 0:
        return images.rot90(int(angle // 90), (-2, -1))
    # Generate every angle from the original. Quarter turns after one residual
    # interpolation preserve the C4 relationship without repeated resampling.
    quarter, residual = divmod(angle, 90)
    radians = math.radians(residual)
    c, s = math.cos(radians), math.sin(radians)
    matrix = images.new_tensor([[c, -s, 0], [s, c, 0]])
    grid = F.affine_grid(matrix[None].expand(len(images), -1, -1),
                         images.shape, align_corners=False)
    rotated = F.grid_sample(images, grid, mode='bilinear',
                            padding_mode='zeros', align_corners=False)
    return rotated.rot90(int(quarter), (-2, -1))


def predict(model, images, fusion):
    if model.invariant:
        return model(images).softmax(1)
    p = model(orbit(images)).softmax(1).reshape(4, len(images), 10)
    if fusion == 'mean':
        return p.mean(0)
    if fusion == 'geometric':
        return p.clamp_min(1e-8).log().mean(0).softmax(1)
    raise ValueError(f'Unsupported frozen fusion: {fusion}')


def score(predictions, labels):
    matrix = np.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10)
    f1 = 2 * np.diag(matrix) / np.maximum(1, matrix.sum(0) + matrix.sum(1))
    return float((predictions == labels).mean()), float(f1.mean())


def bootstrap(delta, draws):
    """Resample paired seeds and original images, keeping all angles together."""
    rng = np.random.default_rng(20261021)
    samples = np.empty(draws)
    for i in range(draws):
        seeds = rng.integers(delta.shape[0], size=delta.shape[0])
        ids = rng.integers(delta.shape[1], size=delta.shape[1])
        samples[i] = delta[seeds][:, ids].mean()
    return {'mean_difference': float(delta.mean()),
            'paired_seed_and_image_bootstrap_95_interval': np.percentile(samples, [2.5, 97.5]).tolist(),
            'draws': draws, 'units': 'paired training seeds and original images; angles kept together'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT.parent / '官方测试_2026-10-11/data/official_test.npz')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/angle_generalization')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--bootstrap-draws', type=int, default=5000)
    parser.add_argument('--plot', action='store_true', help='Also write accuracy_curve.png (requires matplotlib)')
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap_draws < 1:
        parser.error('batch-size and bootstrap-draws must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu or run on the server')
    if args.output.exists():
        parser.error(f'Output already exists; choose a new --output: {args.output}')
    with np.load(args.data, allow_pickle=False) as data:
        images, labels, ids = data['images'], data['labels'], data['ids']
    if images.shape != (10000, 3, 32, 32) or images.dtype != np.uint8:
        raise ValueError('Expected official CIFAR-10 test images: uint8 [10000,3,32,32]')
    if labels.shape != (10000,) or ids.shape != (10000,) or len(np.unique(ids)) != len(ids):
        raise ValueError('Expected 10000 labels and unique original image IDs')
    if not np.issubdtype(labels.dtype, np.integer) or not np.all((labels >= 0) & (labels < 10)):
        raise ValueError('Labels must be integers in [0,9]')
    specs = []
    for mode in ['aug4_large', 'pool_max_large']:
        for seed in [1, 2, 3]:
            folder = ROOT / 'runs' / f'{mode}_s{seed}_e80'
            fusion = json.loads((folder / 'aggregation_selection.json').read_text())['selected'] if mode == 'aug4_large' else 'feature_max'
            if fusion not in ['mean', 'geometric', 'feature_max']:
                raise ValueError(f'Unsupported fusion: {fusion}')
            specs.append({'run': folder.name, 'mode': mode, 'seed': seed, 'fusion': fusion,
                          'checkpoint_sha256': sha256(folder / 'best.pt')})
    torch.set_num_threads(4)
    torch.manual_seed(20261021)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.2)
    angles = np.arange(0, 360, 15)
    unseen = angles % 90 != 0
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {'status': 'running', 'data': str(args.data.resolve()), 'data_sha256': sha256(args.data),
                'script_sha256': sha256(Path(__file__)), 'experiment_sha256': sha256(ROOT / 'experiment.py'),
                'angles': angles.tolist(), 'seeds': [1, 2, 3], 'models': specs,
                'rotation': 'counterclockwise; bilinear residual in [0,90), then exact quarter turn; black fill; fixed 32x32; rotate before normalization',
                'inference': 'unchanged four-view baseline fusion / four-view feature max; no retraining',
                'scope': 'post-hoc supplementary evaluation on already used official test set; no model selection',
                'device': args.device, 'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                'bootstrap_draws': args.bootstrap_draws}
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    mean = torch.tensor([.4914, .4822, .4465], device=args.device)[None, :, None, None]
    std = torch.tensor([.247, .2435, .2616], device=args.device)[None, :, None, None]
    records, correctness, summaries = [], {}, {}
    for spec in specs:
        checkpoint = torch.load(ROOT / 'runs' / spec['run'] / 'best.pt', map_location='cpu', weights_only=True)
        if checkpoint['config']['mode'] != spec['mode']:
            raise ValueError(f'Checkpoint mode mismatch: {spec["run"]}')
        model = LeNet(spec['mode']).to(args.device).eval()
        model.load_state_dict(checkpoint['model'])
        probabilities = np.empty((len(angles), len(labels), 10), dtype=np.float32)
        with torch.inference_mode():
            for ai, angle in enumerate(angles):
                for start in range(0, len(images), args.batch_size):
                    original = torch.as_tensor(images[start:start + args.batch_size], device=args.device, dtype=torch.float32) / 255
                    batch = (rotate(original, int(angle)) - mean) / std
                    probabilities[ai, start:start + len(batch)] = predict(model, batch, spec['fusion']).cpu().numpy()
                print(f'{spec["run"]}: {angle} degrees complete', flush=True)
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(2), 1, atol=1e-5):
            raise ValueError('Invalid probabilities')
        predictions = probabilities.argmax(2)
        correct = predictions == labels[None]
        correctness.setdefault(spec['mode'], []).append(correct)
        acc = correct.mean(1)
        for ai, angle in enumerate(angles):
            accuracy, f1 = score(predictions[ai], labels)
            records.append({'run': spec['run'], 'seed': spec['seed'], 'method': spec['mode'],
                            'angle': int(angle), 'unseen': bool(unseen[ai]), 'accuracy': accuracy, 'macro_f1': f1})
        summaries[spec['run']] = {'seen_mean_accuracy': float(acc[~unseen].mean()),
                                  'unseen_mean_accuracy': float(acc[unseen].mean()),
                                  'worst_angle_accuracy': float(acc.min()),
                                  'all_angles_mean_accuracy': float(acc.mean())}
        np.savez_compressed(args.output / f'{spec["run"]}.npz', probabilities=probabilities,
                            predictions=predictions, labels=labels, ids=ids, angles=angles)
        del model
    methods = {}
    for mode, values in correctness.items():
        accuracy = np.stack(values).mean(2)
        run_summaries = [summaries[s['run']] for s in specs if s['mode'] == mode]
        methods[mode] = {'angle_accuracy_mean': accuracy.mean(0).tolist(),
                         'angle_accuracy_seed_sd': accuracy.std(0, ddof=1).tolist(),
                         'summary': {key: {'mean': float(np.mean([r[key] for r in run_summaries])),
                                           'seed_sd': float(np.std([r[key] for r in run_summaries], ddof=1))}
                                     for key in run_summaries[0]}}
    delta = (np.stack(correctness['pool_max_large']).astype(float)
             - np.stack(correctness['aug4_large']))
    result = {'angles': angles.tolist(), 'by_run': summaries, 'methods': methods,
              'primary_unseen_comparison': bootstrap(delta[:, unseen].mean(1), args.bootstrap_draws),
              'all_angles_comparison': bootstrap(delta.mean(1), args.bootstrap_draws)}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.output / 'angle_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    protocol['status'] = 'complete'
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        for mode, values in methods.items():
            avg = np.array(values['angle_accuracy_mean']) * 100
            sd = np.array(values['angle_accuracy_seed_sd']) * 100
            ax.plot(angles, avg, marker='o', label=mode)
            ax.fill_between(angles, avg - sd, avg + sd, alpha=.15)
        ax.set(xlabel='Input rotation (degrees)', ylabel='Accuracy (%)', xticks=angles)
        ax.tick_params(axis='x', labelrotation=60)
        ax.legend(title='Mean +/- seed SD')
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(args.output / 'accuracy_curve.png', dpi=180)
        plt.close(fig)
    print(f'Results saved to {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
