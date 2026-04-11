from __future__ import annotations

import argparse
import random
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from coamd.data.datasets import AE_ActionDataset, AE_ActionDataset_kit, Text2MotionDataset, collate_fn
from coamd.evaluation.autoencoder_eval import evaluation_ae_with_clip
from coamd.evaluation.evaluators import Evaluators
from coamd.models.autoencoder import AE
from coamd.utils.config import add_config_argument, parse_with_config
from coamd.utils.train_utils import def_value, print_current_loss, save, update_lr_warm_up


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train the 3-stream autoencoder used by CoAMD.')
    add_config_argument(parser)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='outputs/train_ae/humanml3d')
    parser.add_argument('--dataset_dir', type=str, default='/path/to/datasets')
    parser.add_argument('--dataset_name', type=str, default='t2m', choices=['t2m', 'kit'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--warm_up_iter', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--milestones', nargs='+', type=int, default=[150000, 250000])
    parser.add_argument('--lr_decay', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--recons_loss', type=str, default='l1_smooth', choices=['l1_smooth', 'mse'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_every', type=int, default=100)
    return parser


def main(args: argparse.Namespace) -> None:
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    if args.dataset_name == 't2m':
        data_root = Path(args.dataset_dir) / 'HumanML3D'
        joints_num = 22
        dim_pose = 22 * 3
        annotations_actions = data_root / 'annotations_actions_400.json'
        train_dataset = AE_ActionDataset(
            np.load(data_root / 'mean_std' / 'original_absolute' / 'Mean.npy'),
            np.load(data_root / 'mean_std' / 'original_absolute' / 'Std.npy'),
            str(data_root / 'train.txt'),
            str(data_root / 'new_joints'),
            str(annotations_actions),
            window_size=args.window_size,
        )
        val_dataset = AE_ActionDataset(
            np.load(data_root / 'mean_std' / 'original_absolute' / 'Mean.npy'),
            np.load(data_root / 'mean_std' / 'original_absolute' / 'Std.npy'),
            str(data_root / 'test.txt'),
            str(data_root / 'new_joints'),
            str(annotations_actions),
            window_size=args.window_size,
        )
    else:
        data_root = Path(args.dataset_dir) / 'KIT-ML'
        joints_num = 21
        dim_pose = 21 * 3
        train_dataset = AE_ActionDataset_kit(
            np.load(data_root / 'mean_std' / 'original_absolute' / 'Mean.npy'),
            np.load(data_root / 'mean_std' / 'original_absolute' / 'Std.npy'),
            str(data_root / 'train.txt'),
            str(data_root / 'new_joints'),
            window_size=args.window_size,
        )
        val_dataset = AE_ActionDataset_kit(
            np.load(data_root / 'mean_std' / 'original_absolute' / 'Std.npy') * 0 + np.load(data_root / 'mean_std' / 'original_absolute' / 'Mean.npy'),
            np.load(data_root / 'mean_std' / 'original_absolute' / 'Std.npy'),
            str(data_root / 'test.txt'),
            str(data_root / 'new_joints'),
            window_size=args.window_size,
        )

    mean = np.load(data_root / 'mean_std' / 'original_absolute' / 'Mean.npy')
    std = np.load(data_root / 'mean_std' / 'original_absolute' / 'Std.npy')
    motion_dir = data_root / 'new_joints'
    text_dir = data_root / 'texts'

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers, shuffle=True, pin_memory=True)

    eval_dataset = Text2MotionDataset(mean, std, str(data_root / 'test.txt'), args.dataset_name, str(motion_dir), str(text_dir), 4, 196, 20, evaluation=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32, num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn, shuffle=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AE(input_width=dim_pose, output_emb_width=512, proj_dim=512, down_t=2, stride_t=2, width=512, depth=3, dilation_growth_rate=3, activation='relu', norm=None).to(device)
    logger = SummaryWriter(str(output_dir))
    criterion_rec = torch.nn.SmoothL1Loss() if args.recons_loss == 'l1_smooth' else torch.nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay)

    start_time = time.time()
    total_iters = args.epoch * len(train_loader)
    logs = defaultdict(def_value, OrderedDict())
    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_clip_score, mpjpe = 1000, 0, 0, 0, 0, 100, 0, 100
    eval_wrapper = Evaluators(args.dataset_name, device=device)

    step = 0
    for epoch in range(args.epoch):
        model.train()
        for inner_iter, (motions, _label, _multi_hot) in enumerate(train_loader):
            step += 1
            if step < args.warm_up_iter:
                update_lr_warm_up(step, args.warm_up_iter, optimizer, args.lr)

            motions = motions.to(device)
            gt_joint = motions
            gt_bone, gt_motion = model.modality_generation(gt_joint, model.bones)
            pred_joint, pred_bone, pred_motion = model.forward_3stream(motions)

            loss_rec_joint = criterion_rec(pred_joint, gt_joint)
            loss_rec_bone = criterion_rec(pred_bone, gt_bone)
            loss_rec_motion = criterion_rec(pred_motion[:, 1:, :], gt_motion[:, 1:, :])
            loss = loss_rec_joint + loss_rec_bone + loss_rec_motion

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step >= args.warm_up_iter:
                scheduler.step()

            logs['loss'] += loss.item()
            logs['loss_rec_joint'] += loss_rec_joint.item()
            logs['loss_rec_bone'] += loss_rec_bone.item()
            logs['loss_rec_motion'] += loss_rec_motion.item()
            logs['lr'] += optimizer.param_groups[0]['lr']

            if step % args.log_every == 0:
                mean_loss = OrderedDict()
                for tag, value in logs.items():
                    logger.add_scalar(f'Train/{tag}', value / args.log_every, step)
                    mean_loss[tag] = value / args.log_every
                logs = defaultdict(def_value, OrderedDict())
                print_current_loss(start_time, step, total_iters, mean_loss, epoch=epoch, inner_iter=inner_iter)

        save(str(output_dir / 'latest.tar'), epoch, model, optimizer, scheduler, step, 'ae')

        model.eval()
        val_logs = defaultdict(list)
        with torch.no_grad():
            for motions, _label, _multi_hot in val_loader:
                motions = motions.to(device)
                gt_joint = motions
                gt_bone, gt_motion = model.modality_generation(gt_joint, model.bones)
                pred_joint, pred_bone, pred_motion = model.forward_3stream(motions)
                loss_rec_joint_val = criterion_rec(pred_joint, gt_joint)
                loss_rec_bone_val = criterion_rec(pred_bone, gt_bone)
                loss_rec_motion_val = criterion_rec(pred_motion[:, 1:, :], gt_motion[:, 1:, :])
                val_logs['loss'].append((loss_rec_joint_val + loss_rec_bone_val + loss_rec_motion_val).item())
                val_logs['loss_rec'].append(loss_rec_joint_val.item())

        val_mean_loss = {f'Val/{k}': float(np.mean(v)) for k, v in val_logs.items()}
        for tag, value in val_mean_loss.items():
            logger.add_scalar(tag, value, epoch)

        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_clip_score, mpjpe, _ = evaluation_ae_with_clip(
            str(output_dir), eval_loader, model, logger, epoch, eval_wrapper=eval_wrapper, num_joint=joints_num, device=device,
            best_fid=best_fid, best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
            train_mean=mean, train_std=std, best_matching=best_matching, best_clip_score=best_clip_score,
        )


if __name__ == '__main__':
    main(parse_with_config(build_parser()))
