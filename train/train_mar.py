from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from coamd.data.datasets import Text2Motion_Reward_Action_Dataset, collate_fn
from coamd.evaluation.evaluators import MotionCLIP_Reward_2text_3stream
from coamd.evaluation.metrics import calculate_R_precision
from coamd.evaluation.retrieval_metrics import cosine_similarity_matrix
from coamd.utils.config import add_config_argument, parse_with_config
from coamd.utils.logger import Logger


def save_checkpoint(model: MotionCLIP_Reward_2text_3stream, epoch: int, top1: float, acc: float, path: str) -> None:
    state = {
        'contrast_model': model.state_dict(),
        'ep': epoch,
        'acc': acc,
        'top1': top1,
    }
    model_state_dict = model.state_dict()
    clip_weights = [key for key in model_state_dict.keys() if key.startswith('clip_model.')]
    for key in clip_weights:
        del model_state_dict[key]
    state['contrast_model'] = model_state_dict
    torch.save(state, path)


def load_humanml_label_text_map(repo_root: Path) -> list[str]:
    label_map_path = repo_root / 'assets' / 'label_maps' / 'humanml3d400_label_map.txt'
    with label_map_path.open('r', encoding='utf-8') as infile:
        return [line.strip() for line in infile if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train the MAR reward model for CoAMD.')
    add_config_argument(parser)
    parser.add_argument('--dataset_dir', type=str, default='/path/to/datasets')
    parser.add_argument('--dataset_name', type=str, default='t2m', choices=['t2m'])
    parser.add_argument('--save_dir', type=str, default='outputs/train_mar/humanml3d')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    return parser


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    repo_root = Path(__file__).resolve().parents[1]
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger_writer = Logger(os.path.join(save_dir, 'run.log'))
    logger = SummaryWriter(log_dir=str(save_dir))

    if args.dataset_name != 't2m':
        raise ValueError('The released MAR training pipeline currently targets HumanML3D only.')

    data_root = Path(args.dataset_dir) / 'HumanML3D'
    dim_pose = 22 * 3
    annotations_actions = data_root / 'annotations_actions_400.json'
    humanml3d400_label_text_map = load_humanml_label_text_map(repo_root)

    motion_dir = data_root / 'new_joints'
    text_dir = data_root / 'texts'
    train_split_file = data_root / 'train.txt'
    val_split_file = data_root / 'test.txt'
    mean = np.load(data_root / 'mean_std' / 'original_absolute' / 'Mean.npy')
    std = np.load(data_root / 'mean_std' / 'original_absolute' / 'Std.npy')

    train_dataset = Text2Motion_Reward_Action_Dataset(
        mean, std, str(train_split_file), args.dataset_name, str(motion_dir), str(text_dir),
        str(annotations_actions), humanml3d400_label_text_map, 4, 196, 20, evaluation=False
    )
    val_dataset = Text2Motion_Reward_Action_Dataset(
        mean, std, str(val_split_file), args.dataset_name, str(motion_dir), str(text_dir),
        str(annotations_actions), humanml3d400_label_text_map, 4, 196, 20, evaluation=False
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_fn, drop_last=True
    )

    mar_model = MotionCLIP_Reward_2text_3stream(in_dim=dim_pose).to(device)
    logger_writer.info(mar_model)
    optimizer = optim.AdamW(mar_model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)

    best_acc = 0.0
    best_r_precision = 0.0
    best_matching_score = 0.0
    print('Starting MAR training...')

    for epoch in range(args.epoch):
        mar_model.train()
        train_logs = {'loss': [], 'loss_desc': [], 'loss_recog': [], 'lr': []}
        for batch_data in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epoch} [Train]'):
            captions, motions, m_length, _label, label_text, _multi_hot = batch_data
            motions = motions.to(device).float()

            optimizer.zero_grad()
            loss_desc, loss_recog = mar_model.forward_loss(motions, m_length, captions, category_texts=label_text)
            loss = loss_desc + 0.5 * loss_recog
            loss.backward()
            optimizer.step()

            train_logs['loss'].append(loss.item())
            train_logs['loss_desc'].append(float(loss_desc))
            train_logs['loss_recog'].append(float(loss_recog))
            train_logs['lr'].append(optimizer.param_groups[0]['lr'])

        train_mean_loss = {f'Train/{k}': float(np.mean(v)) for k, v in train_logs.items()}
        for tag, value in train_mean_loss.items():
            logger.add_scalar(tag, value, epoch)
            logger_writer.info(f'epoch [{epoch}] {tag}: {value:.4f}')

        mar_model.eval()
        all_text_embeds = []
        all_motion_embeds = []
        r_precision_real = 0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            label_features = mar_model.encode_text_recognition(humanml3d400_label_text_map).to(device)
            for batch_data in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{args.epoch} [Eval]'):
                captions, motions, m_length, _label, _label_text, multi_hot = batch_data
                motions = motions.to(device).float()
                multi_hot = multi_hot.to(device)

                motion_feat_desc, motion_feat_recog, text_feat_desc, _ = mar_model(motions, m_length, captions)
                logits = cosine_similarity_matrix(motion_feat_recog, label_features)
                preds = logits.argmax(dim=1)
                correct_predictions = multi_hot.gather(1, preds.unsqueeze(1)).squeeze(1)
                total_correct += int(correct_predictions.sum().item())

                temp_r_real = calculate_R_precision(
                    text_feat_desc.cpu().numpy(), motion_feat_desc.cpu().numpy(), top_k=3, sum_all=True
                )
                r_precision_real += temp_r_real
                total_samples += len(motions)
                all_motion_embeds.append(motion_feat_desc.cpu())
                all_text_embeds.append(text_feat_desc.cpu())

        acc = total_correct / total_samples
        all_motion_embeds = torch.cat(all_motion_embeds, dim=0)
        all_text_embeds = torch.cat(all_text_embeds, dim=0)
        r_precision_real = r_precision_real / total_samples
        top1_precision = float(r_precision_real[0])
        sim_matrix = cosine_similarity_matrix(all_motion_embeds, all_text_embeds)
        matching_score = float(torch.mean(torch.diag(sim_matrix)).item())

        logger_writer.info(
            f'Epoch {epoch + 1} Validation Results: ACC={acc:.4f}, '
            f'R-Precision (Top-1, 2, 3): {top1_precision:.4f}, {r_precision_real[1]:.4f}, {r_precision_real[2]:.4f}, '
            f'Matching Score={matching_score:.4f}'
        )
        logger.add_scalar('Eval/Acc', acc, epoch)
        logger.add_scalar('Eval/R_Precision_Top1', top1_precision, epoch)
        logger.add_scalar('Eval/Matching_Score', matching_score, epoch)

        if acc > best_acc:
            best_acc = acc
            save_checkpoint(mar_model, epoch, top1_precision, acc, str(save_dir / 'acc_best.tar'))

        if top1_precision > best_r_precision:
            best_r_precision = top1_precision
            save_checkpoint(mar_model, epoch, top1_precision, acc, str(save_dir / 'top1_best.tar'))

        if matching_score > best_matching_score:
            best_matching_score = matching_score

    logger_writer.info(
        f'Training completed. Best ACC: {best_acc:.4f}, Best R-Precision (Top-1): {best_r_precision:.4f}, '
        f'Best Matching Score: {best_matching_score:.4f}'
    )


if __name__ == '__main__':
    main(parse_with_config(build_parser()))
