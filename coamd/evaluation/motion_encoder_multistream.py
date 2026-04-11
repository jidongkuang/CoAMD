# utils/motion_encoder_multistream.py

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# --- 复制别人的所有模型类到这里 ---
# PositionalEncoding, MS_Emb, Emb_Fusion, ST_TR, BaseEncoder, MultiData
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0., max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # pe = torch.zeros(max_len, 1, d_model)
        # pe[:, 0, 0::2] = torch.sin(position * div_term)
        # pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x) :
        # x = x + self.pe[:x.size(0)]
        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)
# modality-specific embedding
class MS_Emb(nn.Module,):
    def __init__(self, t_input_size, s_input_size, hidden_size) -> None:
        super().__init__()

        self.t_embedding = nn.Sequential(
                            nn.Linear(t_input_size, hidden_size),
                            nn.LayerNorm(hidden_size),
                            nn.ReLU(True),
                            nn.Linear(hidden_size, hidden_size),
        ) 


        self.s_embedding = nn.Sequential(
                            nn.Linear(s_input_size, hidden_size),
                            nn.LayerNorm(hidden_size),
                            nn.ReLU(True),
                            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t_src, s_src):
        t_src = self.t_embedding(t_src)
        s_src = self.s_embedding(s_src)

        return t_src, s_src
    
# fusion module for diffierent modalities
class Emb_Fusion(nn.Module):
    def __init__(self, t_input_size, s_input_size, hidden_size) -> None:
        super().__init__()

        self.t_fusion = nn.Sequential(
                            nn.Linear(t_input_size, hidden_size, bias=False),
        ) 


        self.s_fusion = nn.Sequential(
                            nn.Linear(s_input_size, hidden_size, bias=False),
        )


    def forward(self, t_src, s_src):
        t_src = self.t_fusion(t_src)
        s_src = self.s_fusion(s_src)

        return t_src, s_src
    
# spatio-temporal transformer encoder
class ST_TR(nn.Module):
    def __init__(self, hidden_size, num_head, num_layer) -> None:
        super().__init__()
        self.d_model  = hidden_size 

        self.pe = PositionalEncoding(hidden_size)
        self.spe = torch.nn.Parameter(torch.zeros(1, 22, hidden_size))
        t_layer = TransformerEncoderLayer(self.d_model , num_head, self.d_model , batch_first = True, dropout=0.) 
        self.t_tr = TransformerEncoder(t_layer, num_layer)
        self.t_tr1 = TransformerEncoder(t_layer, num_layer)

        s_layer = TransformerEncoderLayer(self.d_model , num_head, self.d_model , batch_first = True, dropout=0.)
        self.s_tr = TransformerEncoder(s_layer, num_layer)
        self.s_tr1 = TransformerEncoder(s_layer, num_layer)


    def forward(self, t_src, s_src, time_padding_mask=None):
        t_psrc = self.pe(t_src)
        t_out = self.t_tr(t_psrc, src_key_padding_mask=time_padding_mask)
        t_out = self.t_tr1(self.pe(t_out)+t_src, src_key_padding_mask=time_padding_mask)
        # *** 修改全局池化，使其忽略padding部分 ***
        if time_padding_mask is not None:
            # 将padding位置的值设为负无穷大，这样amax就不会选到它们
            t_out_masked = t_out.masked_fill(time_padding_mask.unsqueeze(-1), -torch.finfo(t_out.dtype).max)
            t_g = t_out_masked.amax(dim=1)
        else:
            t_g = t_out.amax(dim=1)

        B, _, _ = t_src.shape
        s_out = self.s_tr(s_src + self.spe.expand(B,-1,-1))  # [bs, m*v, 1024]
        s_out = self.s_tr1(s_out + self.spe.expand(B,-1,-1) + s_src)
        s_g = s_out.amax(dim=1)

        out = torch.cat([t_g,s_g], dim=1)    
        return out, t_g, s_g, t_out, s_out

class BaseEncoder(nn.Module):
    def __init__(self, t_input_size, s_input_size, 
                 hidden_size, num_head, num_layer,
                 ) -> None:
        super().__init__()

        # modality-specific embedding
        self.j_emb = MS_Emb(t_input_size, s_input_size, hidden_size)
        self.b_emb = MS_Emb(t_input_size, s_input_size, hidden_size)
        self.m_emb = MS_Emb(t_input_size, s_input_size, hidden_size)

        # fusion module for diffierent modalities
        self.mm_fusion = Emb_Fusion(hidden_size, hidden_size, hidden_size)
        
        # modality-agnostic encoder
        self.ma_encoder = ST_TR(hidden_size, num_head, num_layer)

    def forward(self, jt, js, bt, bs, mt, ms, time_padding_mask=None):
        # uni-modal feature extraction
        # embedding
        jt_src, js_src = self.j_emb(jt,js)
        bt_src, bs_src = self.b_emb(bt,bs)
        mt_src, ms_src = self.m_emb(mt,ms)

        # multi-modal early fusion
        mmt = (jt_src + bt_src + mt_src) / 3
        mms = (js_src + bs_src + ms_src) / 3
        mmt_src, mms_src = self.mm_fusion(mmt,mms)
        
        # encoding  N, hidden*2
        out, t_g, s_g, t_out, s_out = self.ma_encoder(mmt_src, mms_src, time_padding_mask)

        return out, t_g, s_g, t_out, s_out

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor
        
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# MultiData 类
class MultiData(nn.Module):
    def __init__(self, t_input_size, s_input_size, 
                 hidden_size, num_head, num_layer, projector_dim=512
                 ):
        super(MultiData, self).__init__()
  
        self.d_model  = 2*hidden_size

        # self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
        #              (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
        #              (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12),
        #              (26, 4), (27, 26), (28, 26), (29, 27), (30, 28), (31, 2), (32, 32), (33, 33)]
        self.Bone = [(1, 4), (2, 1), (3, 1), (4, 7), (5, 2), (6, 3), (7, 10), (8, 5), (9, 6),
                     (10, 10), (11, 8), (12, 9), (13, 10), (14, 10), (15, 10), (16, 13), (17, 14),
                     (18, 15), (19, 17), (20, 18), (21, 19), (22, 20)]

        
        self.backbone = BaseEncoder(
            t_input_size, s_input_size,
            hidden_size, num_head, num_layer,
        )

        # # joint-aware projector
        # self.j_projector = nn.Sequential(
        #              nn.Linear(self.d_model, self.d_model),
        #              nn.BatchNorm1d(self.d_model),
        #              nn.ReLU(True),
        #              nn.Linear(self.d_model, self.d_model),
        #              nn.BatchNorm1d(self.d_model),
        #              nn.ReLU(True),
        #              nn.Linear(self.d_model, projector_dim),
        #  )
        # self.t_projector = nn.Sequential(
        #              nn.Linear(hidden_size, hidden_size),
        #              nn.BatchNorm1d(hidden_size),
        #              nn.ReLU(True),
        #              nn.Linear(hidden_size, hidden_size),
        #              nn.BatchNorm1d(hidden_size),
        #              nn.ReLU(True),
        #              nn.Linear(hidden_size, projector_dim),
        #  )
                
        # self.s_projector = nn.Sequential(
        #              nn.Linear(hidden_size, hidden_size),
        #              nn.BatchNorm1d(hidden_size),
        #              nn.ReLU(True),
        #              nn.Linear(hidden_size, hidden_size),
        #              nn.BatchNorm1d(hidden_size),
        #              nn.ReLU(True),
        #              nn.Linear(hidden_size, projector_dim),
        #  )
        # self.j_projector = nn.Sequential(
        #              nn.Linear(self.d_model, self.d_model),
        #              nn.BatchNorm1d(self.d_model),
        #              nn.ReLU(True),
        #              nn.Linear(self.d_model, projector_dim),
        #  )
        self.j_projector = nn.Sequential(
                     nn.Linear(self.d_model, self.d_model),
                     nn.LayerNorm(self.d_model),
                     nn.GELU(),
                     nn.Linear(self.d_model, projector_dim)
         )
        self.t_projector = nn.Sequential(
                     nn.LayerNorm(hidden_size),
                     nn.GELU(),
                     nn.Linear(hidden_size, projector_dim),
         )
                
        self.s_projector = nn.Sequential(
                     nn.LayerNorm(hidden_size),
                     nn.GELU(),
                     nn.Linear(hidden_size, projector_dim),
         )
        

        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
        # self.local_loss = Local_loss()
        
    def modality_generation(self, data_input, modality='joint'):
        N, C, T, V, M = data_input.shape
        if modality == 'joint':
            xt = data_input.permute(0, 2, 4, 3, 1)
            xt = xt.reshape(N, T, M*V*C)
            xs = data_input.permute(0, 4, 3, 2, 1)
            xs = xs.reshape(N, M*V, T*C)

        elif modality == 'bone':
            bone = torch.zeros_like(data_input)
            for v1,v2 in self.Bone:
                bone[:,:,:,v1-1,:] = data_input[:,:,:,v1-1,:] - data_input[:,:,:,v2-1,:]
                xt = bone.permute(0, 2, 4, 3, 1)
                xt = xt.reshape(N, T, M*V*C)
                xs = bone.permute(0, 4, 3, 2, 1)
                xs = xs.reshape(N, M*V, T*C)

        elif modality == 'motion':
            motion = torch.zeros_like(data_input) 
            motion[:,:,:-1,:,:] = data_input[:,:,1:,:,:] - data_input[:,:,:-1,:,:]  
            xt = motion.permute(0, 2, 4, 3, 1)
            xt = xt.reshape(N, T, M*V*C)
            xs = motion.permute(0, 4, 3, 2, 1)
            xs = xs.reshape(N, M*V, T*C)

        return xt, xs


    def forward(self, data, time_padding_mask=None):
    
        # uni-modal augmented view
        jt, js = self.modality_generation(data, 'joint')
        bt, bs = self.modality_generation(data, 'bone')
        mt, ms = self.modality_generation(data, 'motion')
        
        # feature encoding (N, 2*hidden)
        j_t_s, t_g, s_g, t_out, s_out = self.backbone(jt, js, bt, bs, mt, ms, time_padding_mask)
        # decomposing multi-modal features
        visual = self.j_projector(j_t_s)
        t_feat = self.t_projector(t_g)
        s_feat = self.s_projector(s_g)

        return visual, t_feat, s_feat, t_out, s_out
# ... (这里省略了所有这些类的完整代码，假设已经从“别人的代码”中复制过来) ...
# 注意：MultiData 内部的 modality_generation 方法需要从 (B, T, D_pose) 输入适配
# 我们将在下面的封装类中处理这个问题。
class Contrastive_loss_3(nn.Module):
    def __init__(self, tau = 0.4):
        super(Contrastive_loss_3, self).__init__()
        self.tau = tau
    
    def sim(self, z1, z2): 
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t()) 
    
    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        # refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / between_sim.sum(1))
    
    def forward(self, z1, z2, mean = True):
        l1 = self.semi_loss(z1, z2)
        l2 = self.semi_loss(z2, z1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret
        return ret
# # --- 创建一个新的封装类，作为我们的 Motion Encoder ---
# class MultiStreamMotionEncoder(nn.Module):
#     def __init__(self, in_dim, hidden_size=1024, num_head=8, num_layer=1):
#         super().__init__()

        
#         self.in_dim = in_dim
#         self.num_joints = in_dim // 3
        
#         # 假设最多有2个person (M=2), 关节数V需要适配
#         # 别人的V是33，我们的V是22。我们需要填充到33
#         self.num_joints_padded = 22
        
#         # 动态计算输入维度
#         # t_input_size: M*V_pad*C = 22 * 3 = 66
#         # s_input_size: T*C = 196 * 3 = 588 (T是可变的，但 MultiData 内部没有用到 T)
#         # s_input_size 应该也是 M*V_pad*C
#         t_input_size = 22 * 3
#         s_input_size = 196 * 3 # 这里的 196 需要根据实际情况调整或让模型不依赖它

#         # 实际上，MultiData 的 MS_Emb 的 t_input_size 是 M*V*C, s_input_size 是 T*C
#         # 这是为了分别处理 (B, T, M*V*C) 和 (B, M*V, T*C) 形式的数据
        
#         # 重新审视别人的代码：MultiData 的输入是 (N, C, T, V, M)
#         # 我们的输入是 (B, T, D)，其中 D=J*C
        
#         self.backbone = MultiData(
#             t_input_size=t_input_size, # M*V_pad*C = 2*33*3
#             s_input_size=s_input_size, # T*C = 64*3
#             hidden_size=hidden_size,
#             num_head=num_head,
#             num_layer=num_layer
#         )
#         # 定义Contrastive Loss
#         self.loss_fn = Contrastive_loss_3()

#     def forward(self, motion, m_lens, text_features):
#         # motion shape: (B, T, D), D = J*C
#         B, T, D = motion.shape
#         C = 3
#         J = D // C
        
#         # --- 适配输入形状 ---
#         # 1. Reshape to (B, T, J, C)
#         x = motion.view(B, T, J, C)
#         # # 2. Pad joints to 33
#         # pad_size = self.num_joints_padded - J
#         # if pad_size > 0:
#         #     padding = torch.zeros(B, T, pad_size, C, device=motion.device)
#         #     x = torch.cat([x, padding], dim=2)
#         # 3. Add person dimension M=2 (这里我们假设M=1，然后复制一份来模拟M=2)
#         # (B, T, V_pad, C) -> (B, T, V_pad, C, M)
#         x = x.unsqueeze(-1) # shape: (B, T, V, C, 1)
#         # padding_person = torch.zeros_like(x)
#         # x = torch.cat([x, padding_person], dim=-1) # shape: (B, T, 33, 3, 2)
#         # 4. Permute to (B, C, T, V, M)
#         x = x.permute(0, 3, 1, 2, 4)
        
#         # --- 调用别人的模型 ---
#         # 假设 text_features 已经编码好
#         visual, _, visual_t, visual_s, t_out, s_out = self.backbone(x, text_features)

#         # --- 计算多层次损失 ---
#         # 1. Local loss
#         visual_t_proj = self.backbone.t_projector(visual_t)
#         visual_s_proj = self.backbone.s_projector(visual_s)
#         loss_local = (self.loss_fn(visual_t_proj, text_features) + self.loss_fn(visual_s_proj, text_features)) * 0.5
        
#         # 2. Consistency loss
#         loss_ts_consistency = self.loss_fn(visual_t_proj, visual_s_proj)

#         # 3. Temporal segments loss
#         num_segments = 4
#         loss_temporal_segments = 0
#         if T >= num_segments:
#             segment_len = T // num_segments
#             for i in range(num_segments):
#                 t_segment = t_out[:, i*segment_len : (i+1)*segment_len, :].amax(dim=1)
#                 projected_t_segment = self.backbone.t_projector(t_segment)
#                 loss_temporal_segments += self.loss_fn(projected_t_segment, text_features)
#             loss_temporal_segments /= num_segments

#         # 4. Spatial parts loss
#         loss_spatial_parts = 0
#         part_indices = {
#                 "head": [9, 12, 15],
#                 "arm": [13, 14, 16, 17, 18, 19, 20, 21],
#                 "spine": [0, 3, 6],
#                 "leg": [1, 2, 4, 5, 7, 8, 10, 11],}
#         num_persons = x.shape[4]
#         for m in range(num_persons):
#             for part_name, indices in part_indices.items():
#                 # 过滤掉超出我们关节数的索引
#                 valid_indices = [idx for idx in indices if idx < self.num_joints_padded]
#                 if not valid_indices: continue
                
#                 s_part = s_out[:, [idx + m*self.num_joints_padded for idx in valid_indices], :].amax(dim=1)
#                 projected_s_part = self.backbone.s_projector(s_part)
#                 loss_spatial_parts += self.loss_fn(projected_s_part, text_features)
#         loss_spatial_parts /= (len(part_indices) * num_persons) if len(part_indices) > 0 else 1.0

#         # 5. Global loss
#         loss_global = self.loss_fn(visual, text_features)

#         # 6. Total Loss
#         loss_total = loss_global + loss_local + 0.2 * loss_ts_consistency + 0.5 * (loss_temporal_segments + loss_spatial_parts)
        
#         return loss_total, visual
    
class MultiStreamFeatureExtractor(nn.Module):
    def __init__(self, in_dim, hidden_size=1024, proj_dim = 512, num_head=2, num_layer=4):
        super().__init__()
        
        # 别人的模型作为骨干
        self.backbone = MultiData(
            t_input_size=22*3, s_input_size=196*3, 
            hidden_size=hidden_size, num_head=num_head, num_layer=num_layer, projector_dim=proj_dim
        )

    def adapt_input(self, motion):
        """辅助函数，用于将输入适配为(B, C, T, V, M)"""
        B, T, D = motion.shape
        C = 3
        J = D // C
        
        x = motion.view(B, T, J, C)
        
        x = x.unsqueeze(-1)
        
        x = x.permute(0, 3, 1, 2, 4)
        return x

    def forward(self, motion, m_lens=None):
        x = self.adapt_input(motion)
        # 只进行编码，返回所有有用的特征
        # 注意：这里的 text=None，因为我们只想提取视觉特征
        if m_lens is not None:
            T = x.shape[2] 
            # m_lens 是原始长度，padding_mask 也应基于原始长度
            # 但我们的输入 motion 已经是固定长度了，所以 m_lens 对应的是有效长度
            time_padding_mask = torch.arange(T)[None, :] >= m_lens[:, None]
            time_padding_mask = time_padding_mask.to(x.device)
        else:
            time_padding_mask = None
        j_t_s, t_g, s_g, t_out, s_out = self.backbone(x, time_padding_mask=time_padding_mask)
        return j_t_s, t_g, s_g, t_out, s_out
