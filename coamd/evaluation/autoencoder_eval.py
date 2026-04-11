from __future__ import annotations

import os

import torch

from coamd.evaluation.metrics import (
    calculate_R_precision,
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_mpjpe,
    euclidean_distance_matrix,
)
from coamd.utils.motion_process import recover_from_ric
from tqdm import tqdm


@torch.no_grad()
def evaluation_ae_with_clip(
    out_dir,
    val_loader,
    net,
    writer,
    ep,
    eval_wrapper,
    num_joint,
    device,
    best_fid=1000,
    best_div=0,
    best_top1=0,
    best_top2=0,
    best_top3=0,
    best_matching=100,
    best_clip_score=0.0,
    train_mean=None,
    train_std=None,
    save=True,
    draw=True,
):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []
    r_precision_real = 0
    r_precision = 0
    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    clip_score_real_sum = 0
    clip_score_pred_sum = 0
    mpjpe = 0
    num_poses = 0

    for batch in tqdm(val_loader, desc=f'Evaluation Epoch {ep}'):
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch
        motion = motion.to(device)
        (et, em), (et_clip, em_clip) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, caption, motion.clone(), m_length)
        bs = motion.shape[0]

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        pred_pose_eval = net(motion).to(device)
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())

        (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, caption, pred_pose_eval.to(device), m_length
        )
        for i in range(bs):
            gt = torch.from_numpy(bgt[i, : m_length[i]]).float()
            pred = torch.from_numpy(bpred[i, : m_length[i]]).float()
            try:
                gt = recover_from_ric(gt, num_joint)
                pred = recover_from_ric(pred, num_joint)
            except Exception:
                pass
            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            num_poses += gt.shape[0]

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_r = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        r_precision_real += temp_r
        matching_score_real += temp_match

        temp_r = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        r_precision += temp_r
        matching_score_pred += temp_match

        for j in range(bs):
            clip_score_real_sum += (em_clip[j] @ et_clip[j].T).item()
            clip_score_pred_sum += (em_pred_clip[j] @ et_pred_clip[j].T).item()

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    r_precision_real = r_precision_real / nb_sample
    r_precision = r_precision / nb_sample
    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    clip_score_real_avg = clip_score_real_sum / nb_sample
    clip_score_pred_avg = clip_score_pred_sum / nb_sample
    mpjpe = mpjpe / num_poses
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        '--> 	 Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, '
        'R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), '
        'matching_real. %.4f, matching_pred. %.4f, clip_score_real. %.4f, clip_score_pred. %.4f, MPJPE. %.4f'
        % (
            ep, fid, diversity_real, diversity, r_precision_real[0], r_precision_real[1], r_precision_real[2],
            r_precision[0], r_precision[1], r_precision[2], matching_score_real, matching_score_pred,
            clip_score_real_avg, clip_score_pred_avg, mpjpe,
        )
    )
    print(msg)
    if draw and writer is not None:
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity, ep)
        writer.add_scalar('./Test/top1', r_precision[0], ep)
        writer.add_scalar('./Test/top2', r_precision[1], ep)
        writer.add_scalar('./Test/top3', r_precision[2], ep)
        writer.add_scalar('./Test/matching_score', matching_score_pred, ep)
        writer.add_scalar('./Test/CLIP_Score', clip_score_pred_avg, ep)

    if fid < best_fid:
        best_fid = fid
        if save:
            torch.save({'ae': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_fid.tar'))
    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        best_div = diversity
    if r_precision[0] > best_top1:
        best_top1 = r_precision[0]
    if r_precision[1] > best_top2:
        best_top2 = r_precision[1]
    if r_precision[2] > best_top3:
        best_top3 = r_precision[2]
    if matching_score_pred < best_matching:
        best_matching = matching_score_pred
    if clip_score_pred_avg > best_clip_score:
        best_clip_score = clip_score_pred_avg

    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_clip_score, mpjpe, writer
