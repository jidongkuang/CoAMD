#!/usr/bin/env python3
"""Benchmark the existing HumanML3D sampler, without changing model code.

Run from the CoAMD repository in the evaluation environment, for example::

    CUDA_VISIBLE_DEVICES=<RTX-4090-UUID> python benchmark_inference.py \
        --config configs/humanml3d/eval_coamd_guided.yaml --cuda 0

Override the checkpoint/data paths with the same flags as eval_coamd_guided.py.
An optional original MARDM checkout can be supplied with --mardm_root. Its
default checkpoints are checkpoints/t2m/{MARDM_SiT_XL,AE}/model/latest.tar.
Without a complete original MARDM implementation and checkpoints it is skipped;
the CoAMD registry's MARDM aliases are deliberately not used as a baseline.

Latency includes CLIP text encoding, autoregressive/ODE sampling, any MAR
guidance, and final AE decoding. Data I/O, model loading, and metric evaluation
are excluded. At batch size 32, s/motion is synchronized batch time divided by
32: a throughput-equivalent cost, not the latency of an individual request.
The reported SD is the sample SD across these normalized batch measurements.
ODE counters are pass-through observers; solver arguments and outputs are
unchanged. For adaptive dopri5, output points, accepted/rejected steps, and
function evaluations (NFE) are reported separately.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from coamd.utils.config import parse_with_config


ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=str(ROOT / 'configs/humanml3d/eval_coamd_guided.yaml'))
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dataset_name', default='t2m', choices=['t2m'])
    parser.add_argument('--dataset_dir', default='datasets')
    parser.add_argument('--generator_checkpoint', default='pretrained/coamd/best_fid.tar')
    parser.add_argument('--ae_checkpoint', default='pretrained/ae/latest.tar')
    parser.add_argument('--reward_model_path', default='pretrained/mar/top1_best.tar')
    parser.add_argument('--model', default='CoAMD-SiT-XL')
    parser.add_argument('--coamd_depth', type=int, default=6)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--time_steps', type=int, default=12)
    parser.add_argument('--cfg', type=float, default=4.5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--hard_pseudo_reorder', action='store_true')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 32])
    parser.add_argument('--num_prompts', type=int, default=256)
    parser.add_argument('--warmup', type=int, default=10, help='Warmup batch generations per setting and batch size (>=10).')
    parser.add_argument('--output', default='inference_benchmark.json')
    parser.add_argument('--mardm_root', default=None, help='Optional original MARDM checkout; otherwise search this repository.')
    parser.add_argument('--mardm_checkpoint', default=None)
    parser.add_argument('--mardm_ae_checkpoint', default=None)
    return parser


def parse_args():
    parser = build_parser()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', default=None)
    config_args, _ = config_parser.parse_known_args()
    # The shared helper reads explicitly supplied configs. Also read the
    # advertised default when --config is omitted, retaining CLI precedence.
    if config_args.config is None:
        with Path(parser.get_default('config')).open(encoding='utf-8') as stream:
            parser.set_defaults(**(yaml.safe_load(stream) or {}))
    return parse_with_config(parser)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def checked_path(value, label):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'{label}: {path}')
    return path


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def revision(root):
    try:
        return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_weights(model, path, key, allow_missing_clip=False):
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint[key]
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [name for name in missing if not (allow_missing_clip and name.startswith('clip_model.'))]
    if bad_missing or unexpected:
        raise RuntimeError(f'Checkpoint mismatch in {path}: missing={bad_missing}, unexpected={unexpected}')
    del state, checkpoint
    return {'key': key, 'missing_pretrained_clip_keys': len(missing), **file_identity(path)}


def parameter_counts(model):
    if model is None:
        return {'total': 0, 'trainable': 0, 'excluding_clip_model': 0}
    clip_model = getattr(model, 'clip_model', None)
    # MAR also registers CLIP's positional embedding at the root. Deduplicate
    # by parameter identity so that this shared alias is still excluded.
    clip_parameter_ids = {id(p) for p in clip_model.parameters()} if clip_model is not None else set()
    return {
        'total': sum(p.numel() for p in model.parameters()),
        'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'excluding_clip_model': sum(p.numel() for p in model.parameters() if id(p) not in clip_parameter_ids),
    }


def select_prompts(args):
    from coamd.data.datasets import Text2MotionDataset

    data_root = Path(args.dataset_dir).expanduser().resolve() / 'HumanML3D'
    split = checked_path(data_root / 'test.txt', 'HumanML3D test split')
    mean = np.load(data_root / 'mean_std/original_absolute/Mean.npy')
    std = np.load(data_root / 'mean_std/original_absolute/Std.npy')
    seed_all(args.seed)
    # evaluation=False omits GloVe/metric features. Eligibility, caption sampling,
    # subclips, and length quantization are the same as the evaluation loader.
    dataset = Text2MotionDataset(mean, std, str(split), 't2m', str(data_root / 'new_joints'),
                                str(data_root / 'texts'), 4, 196, 20, evaluation=False)
    multiple = math.lcm(*args.batch_sizes)
    count = math.ceil(args.num_prompts / multiple) * multiple
    if len(dataset) < count:
        raise ValueError(f'Test set has {len(dataset)} eligible entries, but {count} are required.')
    indices = random.Random(args.seed).sample(range(len(dataset)), count)
    prompts = []
    for index in indices:
        caption, _motion, frames = dataset[index]
        prompts.append({'sample_id': dataset.name_list[index], 'caption': caption,
                        'frames': int(frames), 'latent_tokens': int(frames // 4)})
    # Keep precisely the same frozen caption/length pairs for every setting.
    encoded = json.dumps(prompts, ensure_ascii=False, sort_keys=True).encode('utf-8')
    metadata = {'root': str(data_root), 'split': file_identity(split),
                'split_sha256': hashlib.sha256(split.read_bytes()).hexdigest(),
                'eligible_entries': len(dataset), 'requested_prompts': args.num_prompts,
                'timed_prompts': count, 'prompt_sha256': hashlib.sha256(encoded).hexdigest(),
                'mean_frames': statistics.mean(item['frames'] for item in prompts),
                'min_frames': min(item['frames'] for item in prompts),
                'max_frames': max(item['frames'] for item in prompts)}
    return prompts, metadata


def find_baseline(args):
    root = Path(args.mardm_root).expanduser().resolve() if args.mardm_root else ROOT
    generator = Path(args.mardm_checkpoint).expanduser().resolve() if args.mardm_checkpoint else root / 'checkpoints/t2m/MARDM_SiT_XL/model/latest.tar'
    ae = Path(args.mardm_ae_checkpoint).expanduser().resolve() if args.mardm_ae_checkpoint else root / 'checkpoints/t2m/AE/model/latest.tar'
    required = [root / 'models/MARDM.py', root / 'models/AE.py', root / 'evaluation_MARDM.py', generator, ae]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return None, {'setting': 'MARDM', 'status': 'skipped', 'reason': 'Original baseline implementation/configuration or checkpoints not available.', 'missing': missing}
    return {'root': root, 'generator': generator, 'ae': ae}, None


def load_models(args, setting, device, baseline):
    reward = None
    if setting == 'mardm':
        # Use the original implementation and its own AE, never a CoAMD alias.
        sys.path.insert(0, str(baseline['root']))
        module = importlib.import_module('models.MARDM')
        ae_module = importlib.import_module('models.AE')
        if Path(module.__file__).resolve() != baseline['root'] / 'models/MARDM.py':
            raise RuntimeError('A different models.MARDM module is already imported.')
        ae = ae_module.AE_models['AE_Model'](input_width=67)
        generator = module.MARDM_models['MARDM-SiT-XL'](ae_dim=ae.output_emb_width, cond_mode='text')
        sources = {'ae': load_weights(ae, baseline['ae'], 'ae'),
                   'generator': load_weights(generator, baseline['generator'], 'ema_mardm', True),
                   'implementation': str(module.__file__), 'source_revision': revision(baseline['root']),
                   'representation': 'original MARDM 67-dimensional features'}
    else:
        from coamd.models.autoencoder import AE
        from coamd.models.coamd import COAMD_models

        ae = AE(input_width=66, output_emb_width=512, proj_dim=512, down_t=2, stride_t=2,
                width=512, depth=3, dilation_growth_rate=3, activation='relu', norm=None)
        generator = COAMD_models[args.model](ae_dim=ae.output_emb_width, cond_mode='text', num_layers=args.coamd_depth)
        sources = {'ae': load_weights(ae, args.ae_checkpoint, 'ae'),
                   'generator': load_weights(generator, args.generator_checkpoint, 'ema_mardm', True),
                   'implementation': inspect.getfile(type(generator)), 'source_revision': revision(ROOT),
                   'representation': '66-dimensional absolute joint coordinates'}
        if setting == 'guided':
            from coamd.evaluation.evaluators import MotionCLIP_Reward_2text_3stream

            reward = MotionCLIP_Reward_2text_3stream(in_dim=66)
            sources['mar'] = load_weights(reward, args.reward_model_path, 'contrast_model', True)
    sources['autoregressive_layers'] = len(generator.MARTransformer)
    counts = {'generator': parameter_counts(generator), 'mar': parameter_counts(reward), 'ae': parameter_counts(ae)}
    for model in (generator, ae, reward):
        if model is not None:
            model.eval().to(device)
    return generator, ae, reward, counts, sources


class ODEObserver:
    """Count public torchdiffeq callbacks while delegating unchanged to odeint."""

    def __init__(self, generator):
        sampler = generator.DiffMLPs.gen_diffusion
        if not hasattr(sampler, 'sample_ode'):
            raise ValueError('This benchmark expects the existing SiT ODE sampler.')
        signature = inspect.signature(sampler.sample_ode)
        self.settings = {key: value.default for key, value in signature.parameters.items()
                         if value.default is not inspect.Parameter.empty}
        bound_sample = sampler.sample_ode()
        self.module = importlib.import_module(type(bound_sample.__self__).__module__)
        self.calls = []

    def reset(self):
        self.calls = []

    @contextmanager
    def observe(self):
        original = self.module.odeint

        def observed_odeint(func, *args, **kwargs):
            counters = {'nfe': 0, 'attempted_steps': 0, 'accepted_steps': 0, 'rejected_steps': 0}

            def observed_func(*f_args, **f_kwargs):
                counters['nfe'] += 1
                return func(*f_args, **f_kwargs)

            def callback(name):
                def count(*cb_args):
                    counters[name] += 1
                return count

            observed_func.callback_step = callback('attempted_steps')
            observed_func.callback_accept_step = callback('accepted_steps')
            observed_func.callback_reject_step = callback('rejected_steps')
            result = original(observed_func, *args, **kwargs)
            if not counters['attempted_steps']:
                # Older torchdiffeq versions may not expose callback statistics.
                for key in ('attempted_steps', 'accepted_steps', 'rejected_steps'):
                    counters[key] = None
            self.calls.append(counters)
            return result

        with patch.object(self.module, 'odeint', observed_odeint):
            yield self


def summary(values):
    values = list(values)
    return {'mean': statistics.mean(values), 'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values), 'max': max(values)}


def generate_batch(generator, ae, reward, setting, captions, lengths, args):
    kwargs = {'temperature': args.temperature, 'hard_pseudo_reorder': args.hard_pseudo_reorder}
    if setting != 'mardm':
        kwargs.update(reward_model=reward, guidance_lambda=1.0 if setting == 'guided' else 0.0, ae=ae)
    # generate() already scopes torch.enable_grad() to the existing MAR update.
    # inference_mode() would disable the required input gradients and is not used.
    with torch.no_grad():
        latents = generator.generate(captions, lengths, args.time_steps, args.cfg, **kwargs)
        return ae.decode(latents)


def measure(args, setting, batch_size, prompts, generator, ae, reward, device):
    print(f'{setting}, batch size {batch_size}: {args.warmup} warmup generations, '
          f'then {len(prompts)} timed prompts', flush=True)
    batches = [(tuple(item['caption'] for item in prompts[start:start + batch_size]),
                torch.tensor([item['latent_tokens'] for item in prompts[start:start + batch_size]],
                             device=device, dtype=torch.long))
               for start in range(0, len(prompts), batch_size)]
    observer = ODEObserver(generator)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    # A separate warmup seed and an identical timing seed isolate weight-loading
    # and warmup RNG consumption from each measured setting.
    seed_all(args.seed + 1_000_000)
    with observer.observe():
        for step in range(args.warmup):
            captions, lengths = batches[step % len(batches)]
            output = generate_batch(generator, ae, reward, setting, captions, lengths, args)
            del output
        torch.cuda.synchronize(device)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        resident_bytes = torch.cuda.memory_allocated(device)
        peak = resident_bytes
        seed_all(args.seed)
        measurements = []
        for index, (captions, lengths) in enumerate(batches):
            observer.reset()
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = generate_batch(generator, ae, reward, setting, captions, lengths, args)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            batch_peak = torch.cuda.max_memory_allocated(device)
            peak = max(peak, batch_peak)
            # Validation and host copies are outside the timed region.
            finite = bool(torch.isfinite(output).all().item())
            if not finite:
                raise RuntimeError(f'Non-finite generated motion in {setting}, batch {index}.')
            if len(observer.calls) != args.time_steps:
                raise RuntimeError(f'Expected {args.time_steps} ODE calls, observed {len(observer.calls)}.')
            measurements.append({'batch_index': index, 'batch_size': len(captions),
                                 'elapsed_seconds': elapsed, 'seconds_per_motion': elapsed / len(captions),
                                 'peak_gpu_memory_bytes': batch_peak,
                                 'ode_calls': list(observer.calls)})
            del output
            # Exclude validation allocations from the next measured batch peak.
            torch.cuda.reset_peak_memory_stats(device)
            if (index + 1) % 2 == 0 or index + 1 == len(batches):
                print(f'{setting}, batch size {batch_size}: {index + 1}/{len(batches)} batches', flush=True)
    values = [item['seconds_per_motion'] for item in measurements]
    total_seconds = sum(item['elapsed_seconds'] for item in measurements)
    ode_summaries = {}
    for key in ('nfe', 'attempted_steps', 'accepted_steps', 'rejected_steps'):
        totals = [sum(call[key] for call in item['ode_calls']) for item in measurements
                  if all(call[key] is not None for call in item['ode_calls'])]
        ode_summaries[key + '_per_generation_batch'] = summary(totals) if totals else None
    return {'setting': setting, 'gamma': 1.0 if setting == 'guided' else 0.0, 'status': 'completed',
            'batch_size': batch_size, 'num_prompts': len(prompts), 'num_timed_batches': len(batches),
            'warmup_generations': args.warmup, 'seconds_per_motion': summary(values),
            'total_timed_seconds': total_seconds, 'motions_per_second': len(prompts) / total_seconds,
            'peak_gpu_memory_gb': peak / 1e9, 'peak_gpu_memory_bytes': peak,
            'resident_gpu_memory_bytes': resident_bytes, 'autoregressive_steps': args.time_steps,
            'ode_solver': observer.settings, 'ode_statistics': ode_summaries,
            'batches': measurements}


def save_json(path, data):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def print_markdown(report):
    names = {'unguided': 'CoAMD, gamma=0 (no MAR)', 'guided': 'CoAMD + MAR, gamma=1', 'mardm': 'MARDM (original 67-D)'}
    print('\n| Setting | Batch | Prompts | Seconds / motion (mean ± SD) | Peak GPU GB | AR steps | ODE points / solve | Accepted steps / batch | NFE / batch | Generator M | MAR M |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for result in report['results']:
        if result['status'] != 'completed':
            print(f"| {names[result['setting']]} | {result['batch_size']} | — | {result['status']}: {result.get('error', '')} | — | — | — | — | — | — | — |")
            continue
        latency = result['seconds_per_motion']
        counts = report['models'][result['setting']]['parameters']
        accepted = result['ode_statistics']['accepted_steps_per_generation_batch']
        nfe = result['ode_statistics']['nfe_per_generation_batch']
        steps_text = f"{accepted['mean']:.1f} ± {accepted['std']:.1f}" if accepted else 'unavailable'
        print(f"| {names[result['setting']]} | {result['batch_size']} | {result['num_prompts']} | "
              f"{latency['mean']:.6f} ± {latency['std']:.6f} | {result['peak_gpu_memory_gb']:.3f} | "
              f"{result['autoregressive_steps']} | {result['ode_solver'].get('num_steps', '—')} | {steps_text} | "
              f"{nfe['mean']:.1f} ± {nfe['std']:.1f} | {counts['generator']['total'] / 1e6:.3f} | {counts['mar']['total'] / 1e6:.3f} |")
    for item in report['skipped']:
        print(f"\nSkipped {item['setting']}: {item['reason']}")
    print('\nGB = 10^9 bytes. Peak allocated memory includes resident model weights; warmup is excluded.')
    print('SD is the sample SD of synchronized batch time / batch size. Batch-32 cost is throughput-equivalent.')
    print('Adaptive dopri5 output points are not integration steps. Accepted steps and NFE sum all AR solves per batch.')
    print('Generator counts exclude AE and include registered frozen CLIP weights; AE and non-CLIP counts are saved in JSON.')
    print('gamma is passed unchanged as guidance_lambda; the existing sampler applies (0.5 + guidance_lambda) when positive.')


def main(args):
    if args.num_prompts < 200 or args.warmup < 10:
        raise ValueError('Use at least 200 timed prompts and at least 10 warmup generations.')
    if not args.batch_sizes or any(size < 1 for size in args.batch_sizes) or len(set(args.batch_sizes)) != len(args.batch_sizes):
        raise ValueError('batch_sizes must be unique positive integers.')
    if args.dataset_name != 't2m' or args.time_steps < 1 or args.coamd_depth < 1:
        raise ValueError('Use HumanML3D (t2m) and positive sampling/depth settings.')
    if not torch.cuda.is_available():
        raise RuntimeError('An RTX 4090 CUDA device is required.')
    device = torch.device(f'cuda:{args.cuda}')
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    if 'RTX 4090' not in gpu_name:
        raise RuntimeError(f'Required RTX 4090, got {gpu_name}. Select the correct GPU with CUDA_VISIBLE_DEVICES/--cuda.')
    for name in ('generator_checkpoint', 'ae_checkpoint', 'reward_model_path'):
        setattr(args, name, str(checked_path(getattr(args, name), name)))
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(False)
    baseline, skipped = find_baseline(args)
    prompts, dataset = select_prompts(args)
    print(f'{gpu_name}; {len(prompts)} frozen test prompts; batches {args.batch_sizes}; AR={args.time_steps}; CFG={args.cfg}', flush=True)
    report = {'schema_version': 1, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'running', 'configuration': vars(args), 'hardware': {
                  'gpu_name': gpu_name, 'selected_device': str(device), 'devices_used': 1,
                  'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                  'gpu_total_memory_bytes': torch.cuda.get_device_properties(device).total_memory,
                  'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
                  'cudnn_version': torch.backends.cudnn.version(), 'python_version': platform.python_version(),
                  'torch_cpu_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads()},
              'measurement': {'scope': 'text encoding + generation + optional MAR guidance + final AE decode',
                  'excluded': 'data I/O, model/checkpoint loading, metric networks, metric calculation, output saving',
                  'precision': 'existing model precision, no autocast', 'allow_tf32': True,
                  'cudnn_benchmark': False, 'autograd_anomaly_detection': False, 'memory_gb_divisor': 1e9,
                  'cudnn_deterministic': torch.backends.cudnn.deterministic,
                  'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                  'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                  'standard_deviation': 'sample SD across batch wall times divided by actual batch size',
                  'ode_observer': 'pass-through odeint wrapper and public step callbacks; unchanged solver arguments',
                  'gamma_argument': 'guidance_lambda; positive values retain existing internal 0.5 offset'},
              'dataset': dataset, 'prompts': prompts, 'models': {}, 'results': [], 'skipped': [skipped] if skipped else []}
    settings = ['unguided', 'guided'] + (['mardm'] if baseline else [])
    for setting in settings:
        generator = ae = reward = None
        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            seed_all(args.seed)
            print(f'\nLoading {setting}...', flush=True)
            generator, ae, reward, counts, sources = load_models(args, setting, device, baseline)
            report['models'][setting] = {'parameters': counts, 'checkpoints': sources}
            for batch_size in args.batch_sizes:
                try:
                    result = measure(args, setting, batch_size, prompts, generator, ae, reward, device)
                except torch.cuda.OutOfMemoryError as error:
                    result = {'setting': setting, 'batch_size': batch_size, 'status': 'out_of_memory', 'error': str(error)}
                    gc.collect()
                    torch.cuda.empty_cache()
                report['results'].append(result)
                save_json(args.output, report)
        finally:
            del generator, ae, reward
            gc.collect()
            torch.cuda.empty_cache()
    report['status'] = 'completed' if all(item['status'] == 'completed' for item in report['results']) else 'partial'
    report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    save_json(args.output, report)
    print_markdown(report)
    print(f'\nSaved {Path(args.output).resolve()}', flush=True)


if __name__ == '__main__':
    main(parse_args())
