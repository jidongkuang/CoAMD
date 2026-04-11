from __future__ import annotations

import argparse
import copy
import random
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from coamd.data.datasets import Text2MotionDataset, collate_fn
from coamd.evaluation.evaluators import Evaluators
from coamd.evaluation.generation_eval import evaluation_mardm
from coamd.models.autoencoder import AE
from coamd.models.coamd import COAMD_models
from coamd.utils.config import add_config_argument, parse_with_config
from coamd.utils.train_action_utils import def_value, print_current_loss, save, update_ema, update_lr_warm_up


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train the CoAMD generator.')
    add_config_argument(parser)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='outputs/train_coamd/humanml3d')
    parser.add_argument('--ae_checkpoint', type=str, default='pretrained/ae/latest.tar')
    parser.add_argument('--model', type=str, default='CoAMD-SiT-XL')
    parser.add_argument('--dataset_name', type=str, default='t2m', choices=['t2m', 'kit'])
    parser.add_argument('--dataset_dir', type=str, default='/path/to/datasets')
    parser.add_argument('--max_motion_length', type=int, default=196)
    parser.add_argument('--unit_length', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--warm_up_iter', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--milestones', nargs='+', type=int, default=[50000])
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--diffmlps_batch_mul', type=int, default=4)
    parser.add_argument('--need_evaluation', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--is_continue', action='store_true')
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--eval_every_epoch', type=int, default=10)
    parser.add_argument('--eval_start_epoch', type=int, default=400)
    return parser


def main(args: argparse.Namespace) -> None:
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
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
    train_split_file = data_root / 'train.txt'
    val_split_file = data_root / 'val.txt'

    train_dataset = Text2MotionDataset(mean, std, str(train_split_file), args.dataset_name, str(motion_dir), str(text_dir), args.unit_length, args.max_motion_length, 20, evaluation=False)
    val_dataset = Text2MotionDataset(mean, std, str(val_split_file), args.dataset_name, str(motion_dir), str(text_dir), args.unit_length, args.max_motion_length, 20, evaluation=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers, shuffle=True)

    eval_loader = None
    eval_wrapper = None
    if args.need_evaluation:
        eval_split_file = data_root / 'test.txt'
        eval_dataset = Text2MotionDataset(mean, std, str(eval_split_file), args.dataset_name, str(motion_dir), str(text_dir), 4, 196, 20, evaluation=True)
        eval_loader = DataLoader(eval_dataset, batch_size=32, num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn, shuffle=True)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ae = AE(input_width=dim_pose, output_emb_width=512, proj_dim=512, down_t=2, stride_t=2, width=512, depth=3, dilation_growth_rate=3, activation='relu', norm=None)
    ae_ckpt = torch.load(args.ae_checkpoint, map_location='cpu')
    ae.load_state_dict(ae_ckpt['ae'])

    coamd = COAMD_models[args.model](ae_dim=ae.output_emb_width, cond_mode='text')
    ema_coamd = copy.deepcopy(coamd)
    ema_coamd.eval()
    for param in ema_coamd.parameters():
        param.requires_grad_(False)

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    if args.need_evaluation:
        eval_wrapper = Evaluators(args.dataset_name, device=device)

    logger = SummaryWriter(str(save_dir))
    ae.eval().to(device)
    coamd.to(device)
    ema_coamd.to(device)

    optimizer = optim.AdamW(coamd.parameters(), betas=(0.9, 0.99), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay)

    epoch = 0
    total_it = 0
    if args.is_continue:
        latest_ckpt = save_dir / 'latest.tar'
        checkpoint = torch.load(latest_ckpt, map_location=device)
        coamd.load_state_dict(checkpoint['mardm'], strict=False)
        ema_coamd.load_state_dict(checkpoint['ema_mardm'], strict=False)
        epoch = checkpoint['ep']
        total_it = checkpoint['total_it']

    start_time = time.time()
    total_iters = args.epoch * len(train_loader)
    logs = defaultdict(def_value, OrderedDict())
    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, clip_score = 1000, 0, 0, 0, 0, 100, -1
    best_val_loss = float('inf')

    while epoch < args.epoch:
        coamd.train()
        for inner_iter, batch_data in enumerate(train_loader):
            total_it += 1
            if total_it < args.warm_up_iter:
                update_lr_warm_up(total_it, args.warm_up_iter, optimizer, args.lr)

            conds, motion, m_lens = batch_data
            motion = motion.detach().float().to(device)
            m_lens = m_lens.detach().long().to(device)
            latent = ae.encode(motion)
            m_lens = m_lens // 4
            conds = conds.to(device).float() if torch.is_tensor(conds) else conds

            loss = coamd.forward_loss(latent, conds, m_lens)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            logs['loss'] += loss.item()
            logs['lr'] += optimizer.param_groups[0]['lr']
            update_ema(coamd, ema_coamd, 0.9999)

            if total_it % args.log_every == 0:
                mean_loss = OrderedDict()
                for tag, value in logs.items():
                    logger.add_scalar(f'Train/{tag}', value / args.log_every, total_it)
                    mean_loss[tag] = value / args.log_every
                logs = defaultdict(def_value, OrderedDict())
                print_current_loss(start_time, total_it, total_iters, mean_loss, epoch=epoch, inner_iter=inner_iter)

        save(str(save_dir / 'latest.tar'), epoch, coamd, optimizer, scheduler, total_it, 'mardm', ema_mardm=ema_coamd)
        epoch += 1

        coamd.eval()
        val_losses = []
        with torch.no_grad():
            for batch_data in val_loader:
                conds, motion, m_lens = batch_data
                motion = motion.detach().float().to(device)
                m_lens = m_lens.detach().long().to(device)
                latent = ae.encode(motion)
                m_lens = m_lens // 4
                conds = conds.to(device).float() if torch.is_tensor(conds) else conds
                val_losses.append(float(coamd.forward_loss(latent, conds, m_lens).item()))

        val_loss = float(np.mean(val_losses))
        logger.add_scalar('Val/loss', val_loss, epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        should_eval = args.need_evaluation and epoch >= args.eval_start_epoch and epoch % args.eval_every_epoch == 0
        if should_eval and eval_loader is not None and eval_wrapper is not None:
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, _, clip_score, _, save_now = evaluation_mardm(
                str(save_dir), eval_loader, ema_coamd, ae, logger, epoch - 1, best_fid=best_fid, clip_score_old=clip_score,
                best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                best_matching=best_matching, eval_wrapper=eval_wrapper, device=device, train_mean=mean, train_std=std
            )
            if save_now:
                save(str(save_dir / 'best_fid.tar'), epoch - 1, coamd, optimizer, scheduler, total_it, 'mardm', ema_mardm=ema_coamd)


if __name__ == '__main__':
    main(parse_with_config(build_parser()))
