import os
import numpy as np
from scipy import linalg
import torch
from coamd.utils.motion_process import recover_from_ric
from tqdm import tqdm
import torch.nn.functional as F

#################################################################################
#                               Eval Function Loops                             #
#################################################################################
@torch.no_grad()
def evaluation_ae(out_dir, val_loader, net, writer, ep, eval_wrapper, num_joint, device, best_fid=1000, best_div=0,
                  best_top1=0, best_top2=0, best_top3=0, best_matching=100,
                  train_mean=None, train_std=None, save=True, draw=True):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    mpjpe = 0
    num_poses = 0

    for batch in tqdm(val_loader):
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.to(device)
        (et, em), (_, _) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, caption, motion.clone(), m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        # motion = val_loader.dataset.transform(bgt, train_mean, train_std)

        pred_pose_eval = net(motion).to(device)
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        # bpredd = val_loader.dataset.transform(bpred)

        (et_pred, em_pred), (_, _) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, caption,
                                                          pred_pose_eval.to(device), m_length)
        for i in range(bs):
            gt = torch.from_numpy(bgt)
            pred = torch.from_numpy(bpred)
            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            num_poses += gt.shape[0]

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    mpjpe = mpjpe / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, MPJPE. %.4f" % \
          (ep, fid, diversity_real, diversity, R_precision_real[0], R_precision_real[1], R_precision_real[2],
           R_precision[0], R_precision[1], R_precision[2], matching_score_real, matching_score_pred, mpjpe)
    print(msg)
    if draw:
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity, ep)
        writer.add_scalar('./Test/top1', R_precision[0], ep)
        writer.add_scalar('./Test/top2', R_precision[1], ep)
        writer.add_scalar('./Test/top3', R_precision[2], ep)
        writer.add_scalar('./Test/matching_score', matching_score_pred, ep)

    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw: print(msg)
        best_fid = fid
        if save:
            torch.save({'ae': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_fid.tar'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!!"%(best_div, diversity)
        if draw: print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision[0])
        if draw: print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision[1])
        if draw: print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision[2])
        if draw: print(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_pred)
        if draw: print(msg)
        best_matching = matching_score_pred

    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, mpjpe, writer

def evaluation_ae_with_clip(out_dir, val_loader, net, writer, ep, eval_wrapper, num_joint, device, best_fid=1000, best_div=0,
                  best_top1=0, best_top2=0, best_top3=0, best_matching=100, best_clip_score=0.0, 
                  train_mean=None, train_std=None, save=True, draw=True):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0

    # 新增：为 clip_score 初始化累加器
    clip_score_real_sum = 0
    clip_score_pred_sum = 0

    mpjpe = 0
    num_poses = 0

    for batch in tqdm(val_loader, desc=f"Evaluation Epoch {ep}"):
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.to(device)
        (et, em), (et_clip, em_clip) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, caption, motion.clone(), m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        # motion = val_loader.dataset.transform(bgt, train_mean, train_std)

        pred_pose_eval = net(motion).to(device)
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        # bpredd = val_loader.dataset.transform(bpred)

        (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, caption,
                                                          pred_pose_eval.to(device), m_length)
        for i in range(bs):
            gt = torch.from_numpy(bgt)
            pred = torch.from_numpy(bpred)
            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            num_poses += gt.shape[0]

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        # --- 6. 新增：累加 Clip Score ---
        # 遍历batch，计算每个样本的clip score
        for j in range(bs):
            # 真实动作的 clip score
            clip_score_real_sum += (em_clip[j] @ et_clip[j].T).item()
            # 重建动作的 clip score
            clip_score_pred_sum += (em_pred_clip[j] @ et_pred_clip[j].T).item()

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    # 新增：计算平均 clip_score
    clip_score_real_avg = clip_score_real_sum / nb_sample
    clip_score_pred_avg = clip_score_pred_sum / nb_sample
    
    mpjpe = mpjpe / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    # --- 8. 打印和记录日志 (加入 clip_score) ---
    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, clip_score_real. %.4f, clip_score_pred. %.4f, MPJPE. %.4f" % \
          (ep, fid, diversity_real, diversity, R_precision_real[0], R_precision_real[1], R_precision_real[2],
           R_precision[0], R_precision[1], R_precision[2], matching_score_real, matching_score_pred, clip_score_real_avg, clip_score_pred_avg, mpjpe)
    print(msg)
    if draw:
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity, ep)
        writer.add_scalar('./Test/top1', R_precision[0], ep)
        writer.add_scalar('./Test/top2', R_precision[1], ep)
        writer.add_scalar('./Test/top3', R_precision[2], ep)
        writer.add_scalar('./Test/matching_score', matching_score_pred, ep)
        writer.add_scalar('./Test/CLIP_Score', clip_score_pred_avg, ep) # 新增

    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw: print(msg)
        best_fid = fid
        if save:
            torch.save({'ae': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_fid.tar'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!!"%(best_div, diversity)
        if draw: print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision[0])
        if draw: print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision[1])
        if draw: print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision[2])
        if draw: print(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_pred)
        if draw: print(msg)
        best_matching = matching_score_pred

    # 新增：更新最佳 clip_score
    if clip_score_pred_avg > best_clip_score:
        print(f"--> --> \t CLIP Score Improved from {best_clip_score:.4f} to {clip_score_pred_avg:.4f} !!!")
        best_clip_score = clip_score_pred_avg

    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_clip_score, mpjpe, writer

@torch.no_grad()
def evaluation_mardm(out_dir, val_loader, ema_mardm, ae, writer, ep, best_fid, best_div,
                        best_top1, best_top2, best_top3, best_matching, eval_wrapper, device, clip_score_old, time_steps=None,
                        cond_scale=None, temperature=1, cal_mm=False, train_mean=None, train_std=None, plot_func=None,
                        draw=True, hard_pseudo_reorder=False, reward_model=None, guidance_lambda=0.0, f=None, database=None, save_motion=False, save_dir=None):

    ema_mardm.eval()
    ae.eval()

    save=False

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0
    if time_steps is None: time_steps = 18
    if cond_scale is None:
        if "kit" in out_dir:
            cond_scale = 2.5
        else:
            cond_scale = 4.5
    clip_score_real = 0
    clip_score_gt = 0

    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0
    save_motion_list = [] 
    
        
    for i, batch in enumerate(tqdm(val_loader)):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        clip_text = list(clip_text)
        m_length = m_length.to(device)

        bs, seq = pose.shape[:2]
        if i < num_mm_batch:
            motion_multimodality_batch = []
            batch_clip_score_pred = 0
            for _ in tqdm(range(30)):
                pred_latents = ema_mardm.generate(clip_text, m_length//4 , time_steps, cond_scale,
                                                  temperature=temperature, hard_pseudo_reorder=hard_pseudo_reorder, reward_model=reward_model, guidance_lambda=guidance_lambda,ae=ae, database=database)
                pred_motions = ae.decode(pred_latents)
                # pred_motions = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy(), train_mean, train_std)
                # pred_motions = val_loader.dataset.transform(pred_motions)
                (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len,
                                                                            clip_text,
                                                                            pred_motions.to(device),
                                                                            m_length)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
            for j in range(bs):
                single_em = em_pred_clip[j]
                single_et = et_pred_clip[j]
                clip_score = torch.dot(single_em, single_et).item()
                batch_clip_score_pred += clip_score
            clip_score_real += batch_clip_score_pred

        else:
            pred_latents = ema_mardm.generate(clip_text, m_length//4 , time_steps, cond_scale,
                                              temperature=temperature, hard_pseudo_reorder=hard_pseudo_reorder, reward_model=reward_model, guidance_lambda=guidance_lambda, ae=ae, database=database)

            pred_motions = ae.decode(pred_latents)
            if save_motion:
                pred_motions_original = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy())
                # pred_motions_original = torch.from_numpy(pred_motions_original).float().to(device)

                for i in range(len(clip_text)):
                    current_motion = pred_motions_original[i]
                    current_length = m_length[i].item() #m_length[j].item()
                    current_text = clip_text[i]
                    # 截取到实际长度并重塑
                    current_motion = current_motion[:current_length]
                    joints_data = current_motion.reshape(current_length, 22, 3)
                    item = {
                        'joints': joints_data,
                        'text': current_text,
                        'length': current_length,
                        'hint': None
                            }
                    save_motion_list.append(item)

            
            # pred_motions = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy(), train_mean, train_std)
            # pred_motions = val_loader.dataset.transform(pred_motions)
            (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(word_embeddings,
                                                                              pos_one_hots, sent_len,
                                                                              clip_text,
                                                                              pred_motions.to(device),
                                                                              m_length)
            batch_clip_score_pred = 0
            for j in range(bs):
                single_em = em_pred_clip[j]
                single_et = et_pred_clip[j]
                clip_score = torch.dot(single_em, single_et).item()
                batch_clip_score_pred += clip_score
            clip_score_real += batch_clip_score_pred

        pose = pose.to(device).float()
        (et, em), (et_clip, em_clip) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, clip_text,
                                                          pose.clone(), m_length)
        batch_clip_score = 0
        for j in range(bs):
            single_em = em_clip[j]
            single_et = et_clip[j]
            clip_score = torch.dot(single_em, single_et).item()
            batch_clip_score += clip_score
        clip_score_gt += batch_clip_score
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs
    if save_motion:
        import pickle
        with open(os.path.join(save_dir, f'gen_motion_{int(guidance_lambda)}.pkl'), 'wb') as file: 
            pickle.dump(save_motion_list, file)
        print(f'Generated motions are saved to {os.path.join(save_dir, f"gen_motion_{int(guidance_lambda)}.pkl")}')
        print("save motion len: ", len(save_motion_list))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    clip_score_real = clip_score_real / nb_sample
    clip_score_gt = clip_score_gt / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    if cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep/Re {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred} multimodality. {multimodality:.4f} clip score real. {clip_score_gt} clip score. {clip_score_real}"
    print(msg)
    if f is not None:
        print(msg, file=f, flush=True)

    if draw:
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity, ep)
        writer.add_scalar('./Test/top1', R_precision[0], ep)
        writer.add_scalar('./Test/top2', R_precision[1], ep)
        writer.add_scalar('./Test/top3', R_precision[2], ep)
        writer.add_scalar('./Test/matching_score', matching_score_pred, ep)
        writer.add_scalar('./Test/clip_score', clip_score_real, ep)


    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        if draw: print(msg)
        best_fid, best_ep = fid, ep
        save=True


    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        if draw: print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        if draw: print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        if draw: print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        if draw: print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        if draw: print(msg)
        best_top3 = R_precision[2]

    if clip_score_real > clip_score_old:
        msg = f"--> --> \t CLIP-score Improved from {clip_score_old:.4f} to {clip_score_real:.4f} !!!"
        if draw: print(msg)
        clip_score_old = clip_score_real

    if cal_mm:
        return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, multimodality, clip_score_old, writer, save
    else:
        return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, 0, clip_score_old, writer, save

import torch
import numpy as np
from tqdm import tqdm
from coamd.evaluation.metrics import calculate_activation_statistics, calculate_diversity, calculate_frechet_distance, calculate_R_precision, euclidean_distance_matrix, calculate_multimodality
from coamd.utils.train_utils import lengths_to_mask

@torch.no_grad()
def evaluation_mardm_edit(
    val_loader,
    ema_mardm,
    ae,
    eval_wrapper,
    device,
    time_steps,
    cond_scale,
    temperature=1.0,
    task_name=None,
    num_mm_batch=3,
    mm_runs=30,
    reward_model=None,
    guidance_lambda=0.0,
    force_mask=False
):
    """
    动作编辑/生成评估函数，严格、逐行地复现官方代码的评估逻辑。
    """
    ae.eval()
    ema_mardm.eval()
    if reward_model:
        reward_model.eval()

    # --- 1. 初始化累加器 (与官方代码完全一致) ---
    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    
    R_precision = 0
    matching_score_pred = 0
    
    R_precision_real = 0
    matching_score_real = 0

    # (官方代码中包含了clip_score的计算，我们也加入以保持一致)
    clip_score_real = 0
    clip_score_gt = 0

    nb_sample = 0

    # --- 2. 遍历数据加载器 ---
    for i, batch in enumerate(tqdm(val_loader, desc=f"Evaluating Task: {task_name or 'Generation'}")):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        
        pose = pose.to(device).float()
        m_length = m_length.to(device)
        bs = pose.shape[0]

        gt_latents = ae.encode(pose)
        
        edit_mask = None
        if task_name:
            latent_seq_len = gt_latents.shape[2]
            m_lens_latent = m_length // 4
            edit_mask = torch.zeros(bs, latent_seq_len, dtype=torch.bool, device=device)
            quarter_lens_latent = m_lens_latent // 4
            half_lens_latent = m_lens_latent // 2

            for k in range(bs):
                l_latent = m_lens_latent[k].item()
                q_l_latent = quarter_lens_latent[k].item()
                h_l_latent = half_lens_latent[k].item()
                
                if task_name == 'inpainting':
                    edit_mask[k, q_l_latent : l_latent - q_l_latent] = True
                elif task_name == 'outpainting':
                    edit_mask[k, :q_l_latent] = True
                    edit_mask[k, l_latent - q_l_latent : l_latent] = True
                elif task_name == 'prefix':
                    edit_mask[k, h_l_latent : l_latent] = True
                elif task_name == 'suffix':
                    edit_mask[k, :h_l_latent] = True
        
        # --- 3. 生成/编辑 ---
        if i < num_mm_batch:
            motion_multimodality_batch = []
            # (官方代码没有为MM计算clip_score, 但为了逻辑完整我们保留)
            for _ in tqdm(range(mm_runs)):
                if task_name:
                    pred_latents_mm = ema_mardm.edit(clip_text, latents=gt_latents.clone(), m_lens=m_length // 4, timesteps=time_steps, cond_scale=cond_scale, temperature=temperature, force_mask=force_mask, edit_mask=edit_mask, reward_model=reward_model, guidance_lambda=guidance_lambda)
                else:
                    pred_latents_mm = ema_mardm.generate(clip_text, m_length // 4, time_steps, cond_scale, temperature, reward_model=reward_model, guidance_lambda=guidance_lambda)
                
                pred_motions_mm = ae.decode(pred_latents_mm)

                # ** 修正点：使用完整的返回值 **
                (_, em_pred_mm), _ = eval_wrapper.get_co_embeddings(
                    word_embeddings, pos_one_hots, sent_len, clip_text, pred_motions_mm, m_length
                )
                motion_multimodality_batch.append(em_pred_mm.unsqueeze(1))
            
            motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))
            # 使用最后一次生成的结果进行后续计算
            pred_latents = pred_latents_mm
        else:
            if task_name:
                pred_latents = ema_mardm.edit(clip_text, latents=gt_latents.clone(), m_lens=m_length // 4, timesteps=time_steps, cond_scale=cond_scale, temperature=temperature, force_mask=force_mask, edit_mask=edit_mask, reward_model=reward_model, guidance_lambda=guidance_lambda)
            else:
                pred_latents = ema_mardm.generate(clip_text, m_length // 4, time_steps, cond_scale, temperature, reward_model=reward_model, guidance_lambda=guidance_lambda)

        pred_motions = ae.decode(pred_latents)

        # --- 4. 提取特征 (与官方代码完全一致) ---
        (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, clip_text, pred_motions, m_length
        )
        
        # --- 5. 累加 CLIP Score (与官方代码完全一致) ---
        batch_clip_score_pred = 0
        for j in range(bs):
            # 官方代码使用的是内积 @, 假设特征已经归一化
            # 我们使用 torch.dot 以确保
            single_em = F.normalize(em_pred_clip[j], dim=0)
            single_et = F.normalize(et_pred_clip[j], dim=0)
            clip_score = (single_em @ single_et).item()
            batch_clip_score_pred += clip_score
        clip_score_real += batch_clip_score_pred

        # --- 6. 提取并累加真实动作的特征和指标 (与官方代码完全一致) ---
        (et, em), (et_clip, em_clip) = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, clip_text, pose, m_length
        )

        batch_clip_score_gt = 0
        for j in range(bs):
            single_em = F.normalize(em_clip[j], dim=0)
            single_et = F.normalize(et_clip[j], dim=0)
            clip_score = (single_em @ single_et).item()
            batch_clip_score_gt += clip_score
        clip_score_gt += batch_clip_score_gt

        motion_annotation_list.append(em.cpu())
        motion_pred_list.append(em_pred.cpu())
        
        # --- 7. 逐批次累加 R-Precision 和 Matching Score (与官方代码完全一致) ---
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        temp_R_real = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match_real = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R_real
        matching_score_real += temp_match_real
        
        nb_sample += bs

    # --- 8. 在循环外计算最终指标 (与官方代码完全一致) ---
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).numpy()
    
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    multimodality_val = 0
    if num_mm_batch > 0 and len(motion_multimodality) > 0:
        motion_multimodality_np = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality_val = calculate_multimodality(motion_multimodality_np, 10)

    R_precision_avg = R_precision / nb_sample
    matching_score_avg = matching_score_pred / nb_sample
    
    clip_score_avg = clip_score_real / nb_sample
    clip_score_gt_avg = clip_score_gt / nb_sample
    
    metrics = {
        'fid': fid,
        'diversity': diversity,
        'r_precision': R_precision_avg,
        'matching_score': matching_score_avg,
        'multimodality': multimodality_val,
        'clip_score': clip_score_avg,
        'clip_score_gt': clip_score_gt_avg # 参考值
    }
    
    return metrics


def create_edit_mask(task_name, m_length, total_seq_len):
    """
    为批次中的每个样本创建 edit_mask。
    :param task_name: 'inpainting', 'outpainting', etc.
    :param m_length: 一个包含批次中每个动作真实长度的张量 [B]
    :param total_seq_len: 动作张量的总长度 (e.g., 196)
    :return: 布尔掩码张量 [B, total_seq_len], True表示要编辑的区域
    """
    bs = m_length.shape[0]
    device = m_length.device
    edit_mask = torch.zeros(bs, total_seq_len, dtype=torch.bool, device=device)
    
    quarter_lens = m_length // 4
    half_lens = m_length // 2

    for i in range(bs):
        l = m_length[i].item()
        q_l = quarter_lens[i].item()
        h_l = half_lens[i].item()
        
        if task_name == 'inpainting':
            edit_mask[i, q_l : l - q_l] = True
        elif task_name == 'outpainting':
            edit_mask[i, :q_l] = True
            edit_mask[i, l - q_l : l] = True
        elif task_name == 'prefix':
            edit_mask[i, h_l : l] = True
        elif task_name == 'suffix':
            edit_mask[i, :h_l] = True
            
    return edit_mask

@torch.no_grad()
def evaluation_mardm_edit_official(val_loader, ae_model, trans, repeat_id, eval_wrapper,
                                time_steps, cond_scale, temperature, device, gsample=True, force_mask=False,
                                              cal_mm=False, save_motions=False, save_dir=None, edit_task=None):
    trans.eval()
    ae_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    clip_score_real = 0
    clip_score_gt = 0

    nb_sample = 0
    edit_task = None # inpainting, outpainting, prefix, suffix
    force_mask = False

    all_saved_results = []  # 存储所有生成的动作

    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 3

    from tqdm import tqdm
    for i, batch in enumerate(tqdm(val_loader)):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.to(device)
        pose = pose.to(device).float()
        # print("clip_text:",clip_text) #tuple[str, ...]

        with torch.no_grad():
            latents = ae_model.encode(pose)

        bs, seq = pose.shape[:2]

        latent_seq_len = latents.shape[-1]  # 这通常是原序列长度的1/4
        edit_mask = None
        
        if edit_task in ['inpainting', 'outpainting', 'prefix', 'suffix']:
            # 创建编辑掩码，形状为 (bs, latent_seq_len)，注意这里使用latent空间的序列长度
            # 修改：确保 edit_mask 使用正确的数据类型和设备
            edit_mask = torch.zeros(bs, latent_seq_len, dtype=torch.bool, device=device)
            
            if edit_task in ['inpainting', 'outpainting']:
                # inpainting和outpainting使用25%的长度
                preserve_length = (m_length.float() * 0.25).long() 
                
                for k in range(bs):
                    l = preserve_length[k].item()
                    actual_length = min(l, m_length[k].item())
                    
                    # 将原始长度转换为latent空间长度
                    latent_actual_length = actual_length // 4
                    latent_m_length = m_length[k].item() // 4
                    
                    # 确保索引在有效范围内
                    latent_actual_length = min(latent_actual_length, latent_seq_len)
                    latent_m_length = min(latent_m_length, latent_seq_len)
                    
                    if edit_task == 'inpainting' and latent_actual_length > 0:
                        # inpainting: 编辑中间部分，保留开头和结尾
                        start_preserve = latent_actual_length
                        end_preserve = max(0, latent_m_length - latent_actual_length)
                        if start_preserve < end_preserve:
                            edit_mask[k, start_preserve:end_preserve] = True
                        
                    elif edit_task == 'outpainting':
                        # outpainting: 编辑开头和结尾，保留中间
                        start_idx = latent_actual_length
                        end_idx = max(start_idx, latent_m_length - latent_actual_length)
                        
                        # 编辑开头部分
                        if start_idx > 0:
                            edit_mask[k, :start_idx] = True
                        
                        # 编辑结尾部分
                        if end_idx < latent_m_length:
                            edit_mask[k, end_idx:latent_m_length] = True
                        
            elif edit_task in ['prefix', 'suffix']:
                # prefix和suffix使用50%的长度
                half_length = (m_length.float() * 0.5).long() 
                
                for k in range(bs):
                    l_half = half_length[k].item()
                    actual_half = min(l_half, m_length[k].item())
                    
                    # 将原始长度转换为latent空间长度
                    latent_actual_half = actual_half // 4
                    latent_m_length = m_length[k].item() // 4
                    
                    # 确保索引在有效范围内
                    latent_actual_half = min(latent_actual_half, latent_seq_len)
                    latent_m_length = min(latent_m_length, latent_seq_len)
                    
                    if edit_task == 'suffix' and latent_actual_half < latent_m_length:
                        # prefix: 编辑前50%，保留后50%
                        edit_mask[k, :latent_actual_half] = True
                        
                    elif edit_task == 'prefix' and latent_actual_half > 0:
                        # suffix: 编辑后50%，保留前50%
                        edit_mask[k, latent_actual_half:latent_m_length] = True


        # for i in range(mm_batch)
        if i < num_mm_batch:
            motion_multimodality_batch = []
            batch_clip_score_pred = 0
            for _ in tqdm(range(30)):
                if edit_task is not None:
                    pred_latents = trans.edit(clip_text, latents.clone(), m_length // 4,
                                             timesteps=time_steps, cond_scale=cond_scale,
                                             temperature=temperature, force_mask=False, edit_mask=edit_mask.clone())
                else:
                    pred_latents = trans.generate(clip_text, m_length//4 , time_steps, cond_scale,
                                                    temperature=temperature, hard_pseudo_reorder=False)# for ACMDM`s AE:(B, 22*4, T)

                pred_motions = ae_model.decode(pred_latents)
                
                (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len,
                                                                            clip_text,
                                                                            pred_motions.to(device),
                                                                            m_length)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
            for j in range(bs):
                single_em = em_pred_clip[j]
                single_et = et_pred_clip[j]
                clip_score = (single_em @ single_et.T).item()
                batch_clip_score_pred += clip_score
            clip_score_real += batch_clip_score_pred
        else:
            if edit_task is not None:
                pred_latents = trans.edit(clip_text, latents.clone(), m_length // 4,
                                             timesteps=time_steps, cond_scale=cond_scale,
                                             temperature=temperature, force_mask=False, edit_mask=edit_mask.clone())
            else:
                pred_latents = trans.generate(clip_text, m_length // 4, time_steps, cond_scale,
                                    temperature=temperature, hard_pseudo_reorder=False)


            pred_motions = ae_model.decode(pred_latents)

            # 如果需要保存动作数据
            if save_motions:
                pred_motions = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy())
                pred_motions = torch.from_numpy(pred_motions).float().to(device)
                # m_length = [196] * bs  # 假设所有生成的动作长度都是196
                bs = len(clip_text)
                for j in range(1):#bs
                    current_motion = pred_motions[j]
                    current_length = 143 #m_length[j].item()
                    current_text = clip_text[j]
                    
                    # 截取到实际长度并重塑
                    actual_motion = current_motion[:current_length]
                    joints_data = actual_motion.reshape(current_length, 22, 3)
                    
                    res = {
                        'joints': joints_data.detach().cpu().numpy(),
                        'text': current_text,
                        'length': current_length,
                        'hint': None
                    }
                    
                    all_saved_results.append(res)
            # 保存所有生成的动作
            if save_motions and save_dir:
                import pickle
                os.makedirs(save_dir, exist_ok=True)
                for idx, res in enumerate(all_saved_results):
                    save_path = os.path.join(save_dir, f"vs_8.pkl")
                    with open(save_path, 'wb') as f:
                        pickle.dump(res, f)
                print(f"Saved {len(all_saved_results)} motions to {save_dir}")
            # import pdb; pdb.set_trace()
            #================================
            # pred_motions = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy())
            # pred_motionss = []
            # for j in range(bs):
            #     pred_motionss.append(back_process(pred_motions[j], is_mesh= False))
            # pred_motionss = np.stack(pred_motionss, axis=0)
            # pred_motions = val_loader.dataset.transform(pred_motions, eval_mean, eval_std)
            # pred_motions = torch.from_numpy(pred_motions).to(device)
            #================================

            (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(word_embeddings,
                                                                              pos_one_hots, sent_len,
                                                                              clip_text,
                                                                              pred_motions.to(device),
                                                                              m_length)
            batch_clip_score_pred = 0
            for j in range(bs):
                single_em = em_pred_clip[j]
                single_et = et_pred_clip[j]
                clip_score = (single_em @ single_et.T).item()
                batch_clip_score_pred += clip_score
            clip_score_real += batch_clip_score_pred

        pose = pose.to(device).float()
        #=========================
        # pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
        # poses =[]
        # for j in range(bs):
        #     poses.append(back_process(pose[j], is_mesh=False))
        # poses=np.stack(poses, axis=0)
        # pose = val_loader.dataset.transform(poses, eval_mean, eval_std)
        # pose = torch.from_numpy(pose).to(device)
        # pose = pose.cuda().float()
        #=========================
        (et, em), (et_clip, em_clip) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, clip_text,
                                                          pose.clone(), m_length)
        batch_clip_score = 0
        for j in range(bs):
            single_em = em_clip[j]
            single_et = et_clip[j]
            clip_score = (single_em @ single_et.T).item()
            batch_clip_score += clip_score
        clip_score_gt += batch_clip_score
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    clip_score_real = clip_score_real / nb_sample
    clip_score_gt = clip_score_gt / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"clip_score_real. {clip_score_real:.4f}, clip_score_gt. {clip_score_gt:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, clip_score_real, multimodality, msg

#################################################################################
#                                 Util Functions                                #
#################################################################################
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

#################################################################################
#                                     Metrics                                   #
#################################################################################
def calculate_mpjpe(gt_joints, pred_joints):
    """
    gt_joints: num_poses x num_joints(22) x 3
    pred_joints: num_poses x num_joints(22) x 3
    (obtained from recover_from_ric())
    """
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"

    # Align by root (pelvis)
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    # Compute MPJPE
    mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1) # num_poses x num_joints=22
    mpjpe_seq = mpjpe.mean(-1) # num_poses

    return mpjpe_seq

# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists

def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat


def calculate_matching_score(embedding1, embedding2, sum_all=False):
    assert len(embedding1.shape) == 2
    assert embedding1.shape[0] == embedding2.shape[0]
    assert embedding1.shape[1] == embedding2.shape[1]

    dist = linalg.norm(embedding1 - embedding2, axis=1)
    if sum_all:
        return dist.sum(axis=0)
    else:
        return dist



def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def cosine_similarity_matrix(x1, x2):
    """ 计算两个嵌入矩阵之间的余弦相似度矩阵 """
    x1_norm = F.normalize(x1, p=2, dim=1)
    x2_norm = F.normalize(x2, p=2, dim=1)
    # 相似度矩阵
    sim_matrix = torch.mm(x1_norm, x2_norm.t())
    # 距离矩阵 (1 - similarity)
    # dist_matrix = 1 - sim_matrix
    return sim_matrix

def calculate_R_precision_similarity(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = cosine_similarity_matrix(embedding1, embedding2)
    argmax = np.argsort(-dist_mat.cpu().numpy(), axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat