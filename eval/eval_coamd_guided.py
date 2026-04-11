from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from coamd.data.datasets import Text2MotionDataset, collate_fn
from coamd.evaluation.evaluators import Evaluators, MotionCLIP_Reward_2text_3stream
from coamd.evaluation.generation_eval import evaluation_mardm
from coamd.models.autoencoder import AE
from coamd.models.coamd import COAMD_models
from coamd.utils.config import add_config_argument, parse_with_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Evaluate CoAMD with MAR guidance.')
    add_config_argument(parser)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='outputs/eval_guided/humanml3d')
    parser.add_argument('--generator_checkpoint', type=str, default='pretrained/coamd/best_fid.tar')
    parser.add_argument('--ae_checkpoint', type=str, default='pretrained/ae/latest.tar')
    parser.add_argument('--reward_model_path', type=str, default='pretrained/mar/top1_best.tar')
    parser.add_argument('--guidance_lambda', type=float, default=0.0)
    parser.add_argument('--save_motion', action='store_true')
    parser.add_argument('--model', type=str, default='CoAMD-SiT-XL')
    parser.add_argument('--dataset_name', type=str, default='t2m', choices=['t2m', 'kit'])
    parser.add_argument('--dataset_dir', type=str, default='/path/to/datasets')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--time_steps', type=int, default=12)
    parser.add_argument('--cfg', type=float, default=4.5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--cal_mm', action='store_true')
    parser.add_argument('--hard_pseudo_reorder', action='store_true')
    parser.add_argument('--coamd_depth', type=int, default=6)
    parser.add_argument('--repeat_time', type=int, default=20)
    return parser


def main(args: argparse.Namespace) -> None:
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.dataset_name == 't2m':
        data_root = Path(args.dataset_dir) / 'HumanML3D'
        dim_pose = 22 * 3
    else:
        data_root = Path(args.dataset_dir) / 'KIT-ML'
        dim_pose = 21 * 3

    motion_dir = data_root / 'new_joints'
    text_dir = data_root / 'texts'
    mean = np.load(data_root / 'mean_std' / 'original_absolute' / 'Mean.npy')
    std = np.load(data_root / 'mean_std' / 'original_absolute' / 'Std.npy')
    split_file = data_root / 'test.txt'

    eval_dataset = Text2MotionDataset(mean, std, str(split_file), args.dataset_name, str(motion_dir), str(text_dir), 4, 196, 20, evaluation=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32, num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn, shuffle=True)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / 'eval.log'

    reward_model = MotionCLIP_Reward_2text_3stream(in_dim=dim_pose)
    reward_ckpt = torch.load(args.reward_model_path, map_location='cpu')
    reward_model.load_state_dict(reward_ckpt['contrast_model'], strict=False)

    ae = AE(input_width=dim_pose, output_emb_width=512, proj_dim=512, down_t=2, stride_t=2, width=512, depth=3, dilation_growth_rate=3, activation='relu', norm=None)
    ae_ckpt = torch.load(args.ae_checkpoint, map_location='cpu')
    ae.load_state_dict(ae_ckpt['ae'])

    ema_coamd = COAMD_models[args.model](ae_dim=ae.output_emb_width, cond_mode='text', num_layers=args.coamd_depth)
    generator_ckpt = torch.load(args.generator_checkpoint, map_location='cpu')
    ema_coamd.load_state_dict(generator_ckpt['ema_mardm'], strict=False)

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    eval_wrapper = Evaluators(args.dataset_name, device=device)

    ae.eval().to(device)
    ema_coamd.eval().to(device)
    reward_model.eval().to(device)

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    mm = []
    clip_scores = []

    with log_file.open('a', encoding='utf-8') as f:
        print('=' * 80, file=f)
        print('Experiment Arguments', file=f)
        print('=' * 80, file=f)
        for arg in vars(args):
            print(f'{arg}: {getattr(args, arg)}', file=f)
        print('=' * 80, file=f)

        for repeat_id in range(args.repeat_time):
            with torch.no_grad():
                metrics = evaluation_mardm(
                    str(save_dir), eval_loader, ema_coamd, ae, None, repeat_id, best_fid=1000, clip_score_old=-1,
                    best_div=0, best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper,
                    device=device, train_mean=mean, train_std=std, time_steps=args.time_steps, cond_scale=args.cfg,
                    temperature=args.temperature, cal_mm=args.cal_mm, draw=False, hard_pseudo_reorder=args.hard_pseudo_reorder,
                    reward_model=reward_model, guidance_lambda=args.guidance_lambda, f=f, save_motion=args.save_motion, save_dir=str(save_dir)
                )
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_mm, clip_score, _, _ = metrics
            fid.append(best_fid)
            div.append(best_div)
            top1.append(best_top1)
            top2.append(best_top2)
            top3.append(best_top3)
            matching.append(best_matching)
            mm.append(best_mm)
            clip_scores.append(clip_score)

        fid = np.array(fid)
        div = np.array(div)
        top1 = np.array(top1)
        top2 = np.array(top2)
        top3 = np.array(top3)
        matching = np.array(matching)
        mm = np.array(mm)
        clip_scores = np.array(clip_scores)

        msg_final = (
            f"FID: {np.mean(fid):.4f} +/- {np.std(fid) * 1.96 / np.sqrt(args.repeat_time):.3f}\n"
            f"Diversity: {np.mean(div):.3f} +/- {np.std(div) * 1.96 / np.sqrt(args.repeat_time):.3f}\n"
            f"TOP1: {np.mean(top1):.3f} +/- {np.std(top1) * 1.96 / np.sqrt(args.repeat_time):.3f}\n"
            f"TOP2: {np.mean(top2):.3f} +/- {np.std(top2) * 1.96 / np.sqrt(args.repeat_time):.3f}\n"
            f"TOP3: {np.mean(top3):.3f} +/- {np.std(top3) * 1.96 / np.sqrt(args.repeat_time):.3f}\n"
            f"Matching: {np.mean(matching):.3f} +/- {np.std(matching) * 1.96 / np.sqrt(args.repeat_time):.3f}\n"
            f"Multimodality: {np.mean(mm):.3f} +/- {np.std(mm) * 1.96 / np.sqrt(args.repeat_time):.3f}\n"
            f"CLIP-Score: {np.mean(clip_scores):.3f} +/- {np.std(clip_scores) * 1.96 / np.sqrt(args.repeat_time):.3f}\n"
        )
        print(msg_final)
        print(msg_final, file=f)


if __name__ == '__main__':
    main(parse_with_config(build_parser()))
