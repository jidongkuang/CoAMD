from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from coamd.data.datasets import Text2MotionDataset, collate_fn
from coamd.evaluation.autoencoder_eval import evaluation_ae_with_clip
from coamd.evaluation.evaluators import Evaluators
from coamd.models.autoencoder import AE
from coamd.utils.config import add_config_argument, parse_with_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Evaluate the CoAMD autoencoder.')
    add_config_argument(parser)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='outputs/eval_ae/humanml3d')
    parser.add_argument('--ae_checkpoint', type=str, default='pretrained/ae/latest.tar')
    parser.add_argument('--dataset_dir', type=str, default='/path/to/datasets')
    parser.add_argument('--dataset_name', type=str, default='t2m', choices=['t2m', 'kit'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--repeat_time', type=int, default=10)
    return parser


def main(args: argparse.Namespace) -> None:
    if args.dataset_name == 't2m':
        data_root = Path(args.dataset_dir) / 'HumanML3D'
        joints_num = 22
        dim_pose = 22 * 3
    else:
        data_root = Path(args.dataset_dir) / 'KIT-ML'
        joints_num = 21
        dim_pose = 21 * 3

    motion_dir = data_root / 'new_joints'
    text_dir = data_root / 'texts'
    mean = np.load(data_root / 'mean_std' / 'original_absolute' / 'Mean.npy')
    std = np.load(data_root / 'mean_std' / 'original_absolute' / 'Std.npy')
    eval_dataset = Text2MotionDataset(mean, std, str(data_root / 'test.txt'), args.dataset_name, str(motion_dir), str(text_dir), 4, 196, 20, evaluation=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32, num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn, shuffle=True)

    ae = AE(input_width=dim_pose, output_emb_width=512, proj_dim=512, down_t=2, stride_t=2, width=512, depth=3, dilation_growth_rate=3, activation='relu', norm=None)
    ckpt = torch.load(args.ae_checkpoint, map_location='cpu')
    ae.load_state_dict(ckpt['ae'])

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    ae.to(device).eval()
    eval_wrapper = Evaluators(args.dataset_name, device=device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    mae = []
    clip_scores = []

    for i in range(args.repeat_time):
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_clip_score, mpjpe, _ = evaluation_ae_with_clip(
            str(save_dir), eval_loader, ae, None, i, eval_wrapper=eval_wrapper, num_joint=joints_num, device=device, save=False, draw=False
        )
        fid.append(best_fid)
        div.append(best_div)
        top1.append(best_top1)
        top2.append(best_top2)
        top3.append(best_top3)
        matching.append(best_matching)
        mae.append(mpjpe)
        clip_scores.append(best_clip_score)

    print(f'FID: {np.mean(fid):.5f}')
    print(f'Diversity: {np.mean(div):.3f}')
    print(f'TOP1: {np.mean(top1):.3f}')
    print(f'TOP2: {np.mean(top2):.3f}')
    print(f'TOP3: {np.mean(top3):.3f}')
    print(f'Matching: {np.mean(matching):.3f}')
    print(f'MPJPE: {np.mean(mae):.3f}')
    print(f'CLIP-Score: {np.mean(clip_scores):.3f}')


if __name__ == '__main__':
    main(parse_with_config(build_parser()))
