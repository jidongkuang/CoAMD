import os
import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from coamd.utils.glove import POS_enumerator
import clip
# from .motion_encoder_multistream import MultiStreamMotionEncoder, MultiData
#################################################################################
#                                    Evaluators                                 #
#################################################################################
def build_evaluators(dim_pose, dataset_name, dim_movement_enc_hidden, dim_movement_latent, dim_word, dim_pos_ohot, dim_text_hidden,
                     dim_coemb_hidden, dim_motion_hidden, checkpoints_dir, device):
    movement_enc = MovementConvEncoder(dim_pose, dim_movement_enc_hidden, dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=dim_word,
                                  pos_size=dim_pos_ohot,
                                  hidden_size=dim_text_hidden,
                                  output_size=dim_coemb_hidden,
                                  device=device)

    motion_enc = MotionEncoderBiGRUCo(input_size=dim_movement_latent,
                                      hidden_size=dim_motion_hidden,
                                      output_size=dim_coemb_hidden,
                                      device=device)
    contrast_model = MotionCLIP(dim_pose)

    # checkpoint = torch.load("checkpoints/t2m/text_mot_match_absolute/model/with_val_loss/finest.tar",
    #                         map_location=device)
    print("Loading Checkpoints: ", os.path.join(checkpoints_dir, dataset_name,"text_mot_match_absolute/best/text_mot_match/finest.tar"))
    checkpoint = torch.load(os.path.join(checkpoints_dir, dataset_name,"text_mot_match_absolute/best/text_mot_match/finest.tar"),
                            map_location=device)
    checkpoint_clip = torch.load(os.path.join(checkpoints_dir, dataset_name,"text_mot_match_absolute/best/text_mot_match_clip/finest.tar"),
                            map_location=device)
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    contrast_model.load_state_dict(checkpoint_clip['contrast_model'])
    print('Loading Evaluators')
    return text_enc, motion_enc, movement_enc, contrast_model

class Evaluators(object):

    def __init__(self, dataset_name, device):
        if dataset_name == 't2m':
            dim_pose = 22*3
        elif dataset_name == 'kit':
            dim_pose = 21*3
        else:
            raise KeyError('Dataset not Recognized!!!')

        dim_word = 300
        dim_pos_ohot = len(POS_enumerator)
        dim_motion_hidden = 1024
        dim_movement_enc_hidden = 512
        dim_movement_latent = 512
        dim_text_hidden = 512
        dim_coemb_hidden = 512
        checkpoints_dir = 'checkpoints'
        self.unit_length=4

        self.text_encoder, self.motion_encoder, self.movement_encoder, self.contrast_model \
        = build_evaluators(dim_pose, dataset_name, dim_movement_enc_hidden, dim_movement_latent, dim_word,
                            dim_pos_ohot, dim_text_hidden, dim_coemb_hidden, dim_motion_hidden, checkpoints_dir, device)
        self.device = device

        self.text_encoder.to(device)
        self.motion_encoder.to(device)
        self.movement_encoder.to(device)
        self.contrast_model.to(device)

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()
        self.contrast_model.eval()

    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, captions, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            '''clip based'''
            clip_em = self.contrast_model.encode_motion(motions.clone(), m_lens)
            clip_et = self.contrast_model.encode_text(captions)
            clip_em = clip_em / clip_em.norm(dim=1, keepdim=True)
            clip_et = clip_et / clip_et.norm(dim=1, keepdim=True)

            '''original architecture'''
            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            movements = self.movement_encoder(motions).detach()
            m_lens = m_lens // self.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)

            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
        return (text_embedding, motion_embedding), (clip_et, clip_em)

    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()
            '''clip based'''
            clip_em = self.contrast_model.encode_motion(motions.clone(), m_lens)
            clip_em = clip_em / clip_em.norm(dim=1, keepdim=True)

            '''original architecture'''
            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            movements = self.movement_encoder(motions).detach()
            m_lens = m_lens // self.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding, clip_em

#################################################################################
#                                 Inner Architectures                           #
#################################################################################
def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=300):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, pos):
        return self.pe[pos]


class PositionalEncodingCLIP(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncodingCLIP, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[1], :].unsqueeze(0)
        return self.dropout(x)


def no_grad(nets):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = False

def lengths_to_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


class MovementConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)


class MovementConvDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvDecoder, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(input_size, hidden_size, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(hidden_size, output_size, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)

        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)

class TextEncoderBiGRUCo(nn.Module):
    def __init__(self, word_size, pos_size, hidden_size, output_size, device):
        super(TextEncoderBiGRUCo, self).__init__()
        self.device = device

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class MotionEncoderBiGRUCo(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(MotionEncoderBiGRUCo, self).__init__()
        self.device = device

        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        self.input_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(self, inputs, m_lens):
        num_samples = inputs.shape[0]

        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = m_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class MotionEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim, ff_size, num_layers, num_heads, dropout, activation):
        super().__init__()
        self.input_feats = in_dim
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.query_token = nn.Parameter(torch.randn(1, self.latent_dim))

        self.embed_motion = nn.Linear(self.input_feats, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncodingCLIP(self.latent_dim, self.dropout, max_len=2000)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation,)
        self.transformer = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        self.out_ln = nn.LayerNorm(self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 512)


    def forward(self, motion, padding_mask):
        B, T, D  = motion.shape

        x_emb = self.embed_motion(motion)

        emb = torch.cat([self.query_token[torch.zeros(B, dtype=torch.long, device=motion.device)][:,None], x_emb], dim=1)

        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1)

        h = self.sequence_pos_encoder(emb)
        h = h.permute(1, 0, 2)
        h = self.transformer(h, src_key_padding_mask=padding_mask)
        h = h.permute(1, 0, 2)
        h = self.out_ln(h)
        motion_emb = self.out(h[:,0])

        return motion_emb


class MotionCLIP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.motion_encoder = MotionEncoder(in_dim, 512, 1024, 8, 8, 0.2, 'gelu')
        clip_model, _ = clip.load("ViT-B/16", device="cpu", jit=False)
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        no_grad(self.token_embedding)

        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.2,
            activation="gelu",)
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer,
            num_layers=8)
        self.text_ln = nn.LayerNorm(512)
        self.out = nn.Linear(512, 512)

    def encode_motion(self, motion, m_lens):
        seq_len = motion.shape[1]
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        motion_embedding = self.motion_encoder(motion, padding_mask.to(motion.device))
        return motion_embedding

    def encode_text(self, text):
        device = next(self.parameters()).device

        with torch.no_grad():
            text = clip.tokenize(text, truncate=True).to(device)
            x = self.token_embedding(text).float()
            pe_tokens = x + self.positional_embedding.float()
        pe_tokens = pe_tokens.permute(1,0,2)
        out = self.textTransEncoder(pe_tokens)
        out = out.permute(1, 0, 2)
        out = self.text_ln(out)

        out = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        out = self.out(out)
        return out

    def forward(self, motion, m_lens, text):
        motion_features = self.encode_motion(motion, m_lens)
        text_features = self.encode_text(text)

        motion_features = motion_features / motion_features .norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_motion = logit_scale * motion_features @ text_features.t()
        logits_per_text = logits_per_motion.t()
        return logits_per_motion, logits_per_text

    def forward_loss(self, motion, m_lens, text):
        logits_per_motion, logits_per_text = self.forward(motion, m_lens, text)
        labels = torch.arange(len(logits_per_motion)).to(logits_per_motion.device)

        image_loss = F.cross_entropy(logits_per_motion, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)
        loss = (image_loss + text_loss) / 2
        return loss

# --- 新增 Timestep 嵌入模块 ---
# 这个模块将标量的时间步转换为向量表示
class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, latent_dim, bias=True),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        创建 sin/cos 时间步嵌入。
        参数:
          t: 一个形状为 [N] 的张量，包含要编码的时间步。
          dim: 嵌入的维度。
          max_period: 周期函数的最大周期。
        返回:
          一个形状为 [N, dim] 的张量。
        """
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

    
class MotionEncoder_Reward(nn.Module):
    def __init__(self, in_dim, latent_dim, ff_size, num_layers, num_heads, dropout, activation):
        super().__init__()
        self.input_feats = in_dim
        self.latent_dim = latent_dim
        # ... (其他初始化不变) ...
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        
        self.query_token = nn.Parameter(torch.randn(1, self.latent_dim))
        self.embed_motion = nn.Linear(self.input_feats, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncodingCLIP(self.latent_dim, self.dropout, max_len=2000)

        # *** 新增时间步嵌入器 ***
        self.time_embed = TimestepEmbedder(self.latent_dim)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation,)
        self.transformer = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        self.out_ln = nn.LayerNorm(self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 512)

    # *** 修改 forward 函数以接受 timestep ***
    def forward(self, motion, padding_mask, timestep=None):
        B, T, D = motion.shape
        x_emb = self.embed_motion(motion)

        # *** 如果提供了 timestep，则添加时间嵌入 ***
        if timestep is not None:
            time_emb = self.time_embed(timestep) # (B, latent_dim)
            # 将时间嵌入加到每个序列帧的嵌入上
            x_emb = x_emb + time_emb.unsqueeze(1)

        emb = torch.cat([self.query_token.repeat(B, 1, 1), x_emb], dim=1)
        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1)

        h = self.sequence_pos_encoder(emb)
        h = h.permute(1, 0, 2)
        h = self.transformer(h, src_key_padding_mask=padding_mask)
        h = h.permute(1, 0, 2)
        h = self.out_ln(h)
        motion_emb = self.out(h[:, 0])

        return motion_emb


class MotionCLIP_Reward(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.motion_encoder = MotionEncoder_Reward(in_dim, 512, 1024, 8, 8, 0.2, 'gelu')
        # ... (其他初始化不变) ...
        clip_model, _ = clip.load("ViT-B/16", device="cpu", jit=False)
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        no_grad(self.token_embedding)

        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2, activation="gelu",)
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer, num_layers=8)
        self.text_ln = nn.LayerNorm(512)
        self.out = nn.Linear(512, 512)


    # *** 修改 encode_motion 以传递 timestep ***
    def encode_motion(self, motion, m_lens, timestep=None):
        seq_len = motion.shape[1]
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        motion_embedding = self.motion_encoder(motion, padding_mask.to(motion.device), timestep)
        return motion_embedding

    def encode_text(self, text):
        # ... (代码不变) ...
        device = next(self.parameters()).device
        with torch.no_grad():
            text_tokens = clip.tokenize(text, truncate=True).to(device)
            x = self.token_embedding(text_tokens).float()
            pe_tokens = x + self.positional_embedding.float()
        pe_tokens = pe_tokens.permute(1,0,2)
        out = self.textTransEncoder(pe_tokens)
        out = out.permute(1, 0, 2)
        out = self.text_ln(out)
        out = out[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)]
        out = self.out(out)
        return out

    # *** 修改 forward 和 forward_loss 以接受 timestep ***
    def forward(self, motion, m_lens, text, timestep=None):
        motion_features = self.encode_motion(motion, m_lens, timestep)
        text_features = self.encode_text(text)

        motion_features = motion_features / motion_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_motion = logit_scale * motion_features @ text_features.t()
        logits_per_text = logits_per_motion.t()
        return logits_per_motion, logits_per_text

    def forward_loss(self, motion, m_lens, text, timestep=None):
        logits_per_motion, logits_per_text = self.forward(motion, m_lens, text, timestep)
        labels = torch.arange(len(logits_per_motion)).to(logits_per_motion.device)

        image_loss = F.cross_entropy(logits_per_motion, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)
        loss = (image_loss + text_loss) / 2
        return loss
    

class MotionCLIP_Reward_Action(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.motion_encoder = MotionEncoder_Reward(in_dim, 512, 1024, 8, 8, 0.2, 'gelu')
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_recognition = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # --- 文本编码器 (保持不变) ---
        clip_model, _ = clip.load("ViT-B/16", device="cpu", jit=False)
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        no_grad(self.token_embedding)

        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2, activation="gelu",)
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer, num_layers=8)
        self.text_ln = nn.LayerNorm(512)
        self.out = nn.Linear(512, 512)


        # --- 任务特定的双投影头 ---
        
        # 投影头1: 用于对齐描述性文本 (生成/奖励任务)
        self.description_proj_head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 512)
        )
        
        # 投影头2: 用于对齐类别文本 (识别任务)
        self.recognition_proj_head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 512)
        )

    # *** 修改 encode_motion 以传递 timestep ***
    def encode_motion(self, motion, m_lens, timestep=None):
        seq_len = motion.shape[1]
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        motion_embedding = self.motion_encoder(motion, padding_mask.to(motion.device), timestep)
        return motion_embedding

    def encode_text(self, text):
        # ... (代码不变) ...
        device = next(self.parameters()).device
        with torch.no_grad():
            text_tokens = clip.tokenize(text, truncate=True).to(device)
            x = self.token_embedding(text_tokens).float()
            pe_tokens = x + self.positional_embedding.float()
        pe_tokens = pe_tokens.permute(1,0,2)
        out = self.textTransEncoder(pe_tokens)
        out = out.permute(1, 0, 2)
        out = self.text_ln(out)
        out = out[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)]
        out = self.out(out)
        return out
    
    # *** 修改 forward 和 forward_loss 以接受 timestep ***
    def forward(self, motion, m_lens, desc_texts, category_texts=None, timestep=None):
        motion_features = self.encode_motion(motion, m_lens, timestep)
        # 2. 通过各自的投影头得到任务特定的特征
        motion_features_desc = self.description_proj_head(motion_features)
        motion_features_recog = self.recognition_proj_head(motion_features)
        text_features_desc = self.encode_text(desc_texts)
        if category_texts is not None:
            text_features_recog = self.encode_text(category_texts)
        else:
            text_features_recog = None

        return motion_features_desc, motion_features_recog, text_features_desc, text_features_recog

    def forward_loss(self, motion, m_lens, desc_texts, category_texts, timestep=None):
        motion_feat_desc, motion_feat_recog, text_feat_desc, text_feat_recog = self.forward(motion, m_lens, desc_texts, category_texts, timestep)
        # --- 3. 计算损失1: 描述对齐损失 (InfoNCE) ---
        motion_feat_desc = F.normalize(motion_feat_desc, dim=1)
        text_feat_desc = F.normalize(text_feat_desc, dim=1)
        logits_desc = self.logit_scale.exp() * motion_feat_desc @ text_feat_desc.t()
        labels_desc = torch.arange(len(motion)).to(motion.device)
        loss_desc = (F.cross_entropy(logits_desc, labels_desc) + F.cross_entropy(logits_desc.t(), labels_desc)) / 2

        # --- 4. 计算损失2: 识别对齐损失 (InfoNCE) ---
        motion_feat_recog = F.normalize(motion_feat_recog, dim=1)
        text_feat_recog = F.normalize(text_feat_recog, dim=1)
        logits_recog = self.logit_scale_recognition.exp() * motion_feat_recog @ text_feat_recog.t()
        labels_recog = torch.arange(len(motion)).to(motion.device)
        loss_recog = (F.cross_entropy(logits_recog, labels_recog) + F.cross_entropy(logits_recog.t(), labels_recog)) / 2

        # return 0, loss_recog
        return loss_desc, loss_recog
    
class MotionCLIP_Reward_Action_FrozeCLIP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.motion_encoder = MotionEncoder_Reward(in_dim, 512, 1024, 8, 8, 0.2, 'gelu')
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_recognition = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # --- 文本编码器 (保持不变) ---
        clip_model, clip_preprocess = clip.load('ViT-B/32', device='cpu', jit=False)
        clip.model.convert_weights(clip_model)

        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        self.clip_model = clip_model

        # --- 任务特定的双投影头 ---
        
        # 投影头1: 用于对齐描述性文本 (生成/奖励任务)
        self.description_proj_head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 512)
        )
        
        # 投影头2: 用于对齐类别文本 (识别任务)
        self.recognition_proj_head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 512)
        )


    # *** 修改 encode_motion 以传递 timestep ***
    def encode_motion(self, motion, m_lens, timestep=None):
        seq_len = motion.shape[1]
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        motion_embedding = self.motion_encoder(motion, padding_mask.to(motion.device), timestep)
        motion_features_desc = self.description_proj_head(motion_embedding)
        return motion_features_desc
    
    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    # *** 修改 forward 和 forward_loss 以接受 timestep ***
    def forward(self, motion, m_lens, desc_texts, category_texts=None, timestep=None):
        seq_len = motion.shape[1]
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        motion_features = self.motion_encoder(motion, padding_mask.to(motion.device), timestep)
        # 2. 通过各自的投影头得到任务特定的特征
        motion_features_desc = self.description_proj_head(motion_features)
        motion_features_recog = self.recognition_proj_head(motion_features)
        text_features_desc = self.encode_text(desc_texts)
        if category_texts is not None:
            text_features_recog = self.encode_text(category_texts)
        else:
            text_features_recog = None

        return motion_features_desc, motion_features_recog, text_features_desc, text_features_recog

    def forward_loss(self, motion, m_lens, desc_texts, category_texts, timestep=None):
        motion_feat_desc, motion_feat_recog, text_feat_desc, text_feat_recog = self.forward(motion, m_lens, desc_texts, category_texts, timestep)
        # --- 3. 计算损失1: 描述对齐损失 (InfoNCE) ---
        motion_feat_desc = F.normalize(motion_feat_desc, dim=1)
        text_feat_desc = F.normalize(text_feat_desc, dim=1)
        logits_desc = self.logit_scale.exp() * motion_feat_desc @ text_feat_desc.t()
        labels_desc = torch.arange(len(motion)).to(motion.device)
        loss_desc = (F.cross_entropy(logits_desc, labels_desc) + F.cross_entropy(logits_desc.t(), labels_desc)) / 2

        # --- 4. 计算损失2: 识别对齐损失 (InfoNCE) ---
        motion_feat_recog = F.normalize(motion_feat_recog, dim=1)
        text_feat_recog = F.normalize(text_feat_recog, dim=1)
        logits_recog = self.logit_scale_recognition.exp() * motion_feat_recog @ text_feat_recog.t()
        labels_recog = torch.arange(len(motion)).to(motion.device)
        loss_recog = (F.cross_entropy(logits_recog, labels_recog) + F.cross_entropy(logits_recog.t(), labels_recog)) / 2

        # return 0, loss_recog
        return loss_desc, loss_recog

class TextCLIP(torch.nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model.float()

    def forward(self,text):
        return self.model.encode_text(text)
    
# class MotionCLIP_Reward_Action_STTR_Loss(nn.Module):
#     def __init__(self, in_dim, shared_dim=1024, proj_dim=768):
#         super().__init__()
        
#         # --- 使用别人的模型作为动作编码器 ---
#         # 这个新的编码器内部已经包含了 projector 和损失计算逻辑
#         self.motion_encoder_multistream = MultiStreamMotionEncoder(
#             in_dim=in_dim, 
#             hidden_size=shared_dim, 
#             num_head=1, # 根据别人的代码设置
#             num_layer=2   # 根据别人的代码设置
#         )

#         # --- 文本编码器 (保持不变, 但现在主要用于推理和传入新编码器) ---
#         clip_model, _ = clip.load("ViT-L/14@336px", device="cpu", jit=False)
#         self.text_encoder = TextCLIP(clip_model) # 假设 TextCLIP 是一个封装
#         # (或者直接使用 clip_model.encode_text)

#     def encode_motion(self, motion, m_lens=None):
#         """
#         这个函数现在用于推理/评估，返回最终的全局视觉特征。
#         """
#         # 在评估时，我们只需要全局特征，不需要计算损失
#         # 我们需要修改 MultiStreamMotionEncoder 以支持仅推理模式
        
#         # 让我们简化一下，直接在 forward_loss 中处理
#         # 评估时，我们只需要全局特征
#         x = self.adapt_input(motion)
#         visual, _, _, _, _, _ = self.motion_encoder_multistream.backbone(x, None)
#         return visual

#     def encode_text(self, text):
#         """编码文本，可以是描述或类别"""
#         # 使用你原来的文本编码器
#         return self.text_encoder(text)

#     def adapt_input(self, motion):
#         """辅助函数，用于将输入适配为(B, C, T, V, M)"""
#         B, T, D = motion.shape
#         C = 3
#         J = D // C
#         # num_joints_padded = 22 # 匹配别人的模型
        
#         x = motion.view(B, T, J, C)
#         # pad_size = num_joints_padded - J
#         # if pad_size > 0:
#         #     padding = torch.zeros(B, T, pad_size, C, device=motion.device)
#         #     x = torch.cat([x, padding], dim=2)
        
#         x = x.unsqueeze(-1)
#         # padding_person = torch.zeros_like(x)
#         # x = torch.cat([x, padding_person], dim=-1)
#         x = x.permute(0, 3, 1, 2, 4)
#         return x

#     def forward_loss(self, motion, m_lens, desc_texts, category_texts):
#         """
#         这个函数现在是训练的核心。
#         它使用多流编码器计算复杂的层级化损失。
#         """
#         # 适配输入
#         x = self.adapt_input(motion)
        
#         # --- 针对两种任务，编码两种文本 ---
#         # 1. 描述文本特征 (用于奖励模型)
#         text_feat_desc = self.encode_text(desc_texts)
#         # 2. 类别文本特征 (用于识别)
#         text_feat_recog = self.encode_text(category_texts)

#         # --- 调用多流编码器计算损失 ---
#         # 我们需要两个独立的损失计算过程
        
#         # 任务1: 描述对齐损失
#         loss_desc, _ = self.motion_encoder_multistream(motion, m_lens, text_feat_desc)

#         # 任务2: 识别对齐损失
#         loss_recog, visual_recog = self.motion_encoder_multistream(motion, m_lens, text_feat_recog)
        
#         return loss_desc, loss_recog


# utils/evaluators_action.py

from .motion_encoder_multistream import MultiStreamFeatureExtractor
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
    

class InfoNCE_with_filtering:
    def __init__(self, temperature=0.1, threshold_selfsim=0.9):
        self.temperature = temperature
        # 阈值在0-1之间，内部会映射到-1到1的余弦相似度范围
        self.threshold_selfsim = threshold_selfsim
        print(f"InfoNCE_with_filtering initialized with temp={temperature}, threshold={threshold_selfsim}")

    def get_sim_matrix(self, x, y):
        x_logits = F.normalize(x, dim=-1)
        y_logits = F.normalize(y, dim=-1)
        sim_matrix = x_logits @ y_logits.T
        return sim_matrix

    def __call__(self, motion_features, text_features, sent_embs=None):
        """
        计算过滤后的对比损失。
        :param motion_features: 编码后的动作特征 [B, D]
        :param text_features: 编码后的文本特征 [B, D]
        :param sent_embs: 预计算的句子语义嵌入 [B, D_sent]
        """
        device = motion_features.device
        bs = motion_features.shape[0]
        
        # 计算动作和文本之间的相似度矩阵
        sim_matrix = self.get_sim_matrix(motion_features, text_features) / self.temperature

        if sent_embs is not None and self.threshold_selfsim > 0:
            # 将0-1的阈值映射到-1到1的余弦相似度范围
            # e.g., 0.8 -> 0.6; 0.9 -> 0.8
            real_threshold_selfsim = 2 * self.threshold_selfsim - 1
            
            # 使用预计算的句子嵌入来计算句子间的语义相似度
            sent_embs_normalized = F.normalize(sent_embs, dim=-1)
            # selfsim.shape: [B, B]
            selfsim = sent_embs_normalized @ sent_embs_normalized.T
            
            # 创建一个屏蔽对角线的矩阵
            mask_no_diag = ~torch.eye(bs, dtype=torch.bool, device=device)
            
            # 找到那些非对角线（即不同句子）但语义相似度超过阈值的位置
            filtering_mask = (selfsim > real_threshold_selfsim) & mask_no_diag
            
            # 在这些“假负样本”位置，将相似度设为负无穷，使其在softmax中不起作用
            sim_matrix = sim_matrix.masked_fill(filtering_mask, -torch.finfo(sim_matrix.dtype).max)

        # 计算标准的 InfoNCE 损失
        labels = torch.arange(bs, device=device)
        total_loss = (
            F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
        ) / 2

        return total_loss
    
class MotionCLIP_Reward_Action_STTR_Feature(nn.Module):
    def __init__(self, in_dim, shared_dim=2048, proj_dim=512):
        super().__init__()
        
        # 1. 使用新的多流特征提取器作为主干
        self.motion_feature_extractor = MultiStreamFeatureExtractor(in_dim, hidden_size=shared_dim//2, proj_dim=proj_dim)

        # 2. 文本编码器 (不变)
        # --- 文本编码器 (保持不变) ---
        clip_model, _ = clip.load("ViT-B/16", device="cpu", jit=False)
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_recognition = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        no_grad(self.token_embedding)

        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2, activation="gelu",)
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer, num_layers=8)
        self.text_ln = nn.LayerNorm(512)
        self.out = nn.Linear(512, 512)
        
        # 3. 双投影头 (我们的思想)
        # 投影头1: 描述任务 (作用于全局特征 j_t_s)
        self.desc_proj_head = nn.Sequential(nn.LayerNorm(proj_dim), nn.Linear(proj_dim, proj_dim))
        # 投影头2: 识别任务 (也作用于全局特征 j_t_s)
        self.recog_proj_head = nn.Sequential(nn.LayerNorm(proj_dim), nn.Linear(proj_dim, proj_dim))

        self.loss_fn = Contrastive_loss_3() # 使用别人的损失

    def encode_text(self, text):
        # ... (代码不变) ...
        device = next(self.parameters()).device
        with torch.no_grad():
            text_tokens = clip.tokenize(text, truncate=True).to(device)
            x = self.token_embedding(text_tokens).float()
            pe_tokens = x + self.positional_embedding.float()
        pe_tokens = pe_tokens.permute(1,0,2)
        out = self.textTransEncoder(pe_tokens)
        out = out.permute(1, 0, 2)
        out = self.text_ln(out)
        out = out[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)]
        out = self.out(out)
        return out
    
    def encode_motion(self, motion, m_lens=None):
        """  返回描述任务的全局特征 """
        j_t_s, t_g, s_g, t_out, s_out = self.motion_feature_extractor(motion)
        motion_feat_desc = self.desc_proj_head(j_t_s)

        return motion_feat_desc
    
    def forward(self, motion, m_lens, desc_texts, category_texts=None):
        """
        这个函数现在是训练的核心。
        它使用多流编码器计算复杂的层级化特征。
        """
        # 1. 提取所有层级的动作特征
        j_t_s, t_g, s_g, t_out, s_out = self.motion_feature_extractor(motion)
        
        # 2. 编码两种文本
        text_feat_desc = self.encode_text(desc_texts)
        if category_texts is not None:
            text_feat_recog = self.encode_text(category_texts)
        else:
            text_feat_recog = None

        # 3. 通过各自的投影头得到任务特定的特征
        motion_feat_desc = self.desc_proj_head(j_t_s)
        motion_feat_recog = self.recog_proj_head(j_t_s)

        return motion_feat_desc, motion_feat_recog, text_feat_desc, text_feat_recog

    def forward_loss(self, motion, m_lens, desc_texts, category_texts):
        # 1. 提取所有层级的动作特征
        j_t_s, t_g, s_g, t_out, s_out = self.motion_feature_extractor(motion)
        
        # 2. 编码两种文本
        text_feat_desc = self.encode_text(desc_texts)
        text_feat_recog = self.encode_text(category_texts)

        # 3. 计算描述对齐损失 (只用全局特征)
        # 全局损失
        motion_feat_desc = self.desc_proj_head(j_t_s)
        loss_desc_global = self.loss_fn(motion_feat_desc, text_feat_desc)
        # 局部损失
        t_g_proj_desc = self.desc_proj_head(t_g)
        s_g_proj_desc = self.desc_proj_head(s_g)
        loss_desc_local = (self.loss_fn(t_g_proj_desc, text_feat_desc) + self.loss_fn(s_g_proj_desc, text_feat_desc)) * 0.5
        # (这里可以继续添加分段、分块损失，但为了简化，先用这两部分)
        loss_desc = loss_desc_global + loss_desc_local

        
        # 4. 计算识别对齐损失 (使用别人的多层次方法)
        # 全局损失
        motion_feat_recog = self.recog_proj_head(j_t_s)
        loss_recog_global = self.loss_fn(motion_feat_recog, text_feat_recog)
        # 局部损失
        t_g_proj_recog = self.recog_proj_head(t_g)
        s_g_proj_recog = self.recog_proj_head(s_g)
        loss_recog_local = (self.loss_fn(t_g_proj_recog, text_feat_recog) + self.loss_fn(s_g_proj_recog, text_feat_recog)) * 0.5
        # (这里可以继续添加分段、分块损失，但为了简化，先用这两部分)
        loss_recog = loss_recog_global + loss_recog_local

        
        return loss_desc, loss_recog

class MotionCLIP_Reward_Action_STTR_Feature_FrozeCLIP(nn.Module):
    def __init__(self, in_dim, shared_dim=2048, proj_dim=512):
        super().__init__()
        
        # 1. 使用新的多流特征提取器作为主干
        self.motion_feature_extractor = MultiStreamFeatureExtractor(in_dim, hidden_size=shared_dim//2, proj_dim=proj_dim)

        # 2. 文本编码器 (不变)
        # --- 文本编码器 (保持不变) ---
        clip_model, clip_preprocess = clip.load('ViT-B/32', device='cpu', jit=False)
        clip.model.convert_weights(clip_model)

        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        self.clip_model = clip_model
        
        # 3. 双投影头 (我们的思想)
        # 投影头1: 描述任务 (作用于全局特征 j_t_s)
        self.desc_proj_head = nn.Sequential(nn.LayerNorm(proj_dim), nn.Linear(proj_dim, proj_dim))
        # 投影头2: 识别任务 (也作用于全局特征 j_t_s)
        self.recog_proj_head = nn.Sequential(nn.LayerNorm(proj_dim), nn.Linear(proj_dim, proj_dim))

        self.loss_fn = Contrastive_loss_3() # 使用别人的损失

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    
    def encode_motion(self, motion, m_lens=None):
        """  返回描述任务的全局特征 """
        j_t_s, t_g, s_g, t_out, s_out = self.motion_feature_extractor(motion)
        motion_feat_desc = self.desc_proj_head(j_t_s)

        return motion_feat_desc
    
    def forward(self, motion, m_lens, desc_texts, category_texts=None):
        """
        这个函数现在是训练的核心。
        它使用多流编码器计算复杂的层级化特征。
        """
        # 1. 提取所有层级的动作特征
        j_t_s, t_g, s_g, t_out, s_out = self.motion_feature_extractor(motion, m_lens)
        
        # 2. 编码两种文本
        text_feat_desc = self.encode_text(desc_texts)
        if category_texts is not None:
            text_feat_recog = self.encode_text(category_texts)
        else:
            text_feat_recog = None

        # 3. 通过各自的投影头得到任务特定的特征
        motion_feat_desc = self.desc_proj_head(j_t_s)
        motion_feat_recog = self.recog_proj_head(j_t_s)

        return motion_feat_desc, motion_feat_recog, text_feat_desc, text_feat_recog

    def forward_loss(self, motion, m_lens, desc_texts, category_texts):
        # 1. 提取所有层级的动作特征
        j_t_s, t_g, s_g, t_out, s_out = self.motion_feature_extractor(motion, m_lens)
        
        # 2. 编码两种文本
        text_feat_desc = self.encode_text(desc_texts)
        text_feat_recog = self.encode_text(category_texts)

        # 3. 计算描述对齐损失 (只用全局特征)
        # 全局损失
        motion_feat_desc = self.desc_proj_head(j_t_s)
        loss_desc_global = self.loss_fn(motion_feat_desc, text_feat_desc)
        # 局部损失
        t_g_proj_desc = self.desc_proj_head(t_g)
        s_g_proj_desc = self.desc_proj_head(s_g)
        loss_desc_local = (self.loss_fn(t_g_proj_desc, text_feat_desc) + self.loss_fn(s_g_proj_desc, text_feat_desc)) * 0.5
        # (这里可以继续添加分段、分块损失，但为了简化，先用这两部分)
        loss_desc = loss_desc_global + loss_desc_local

        
        # 4. 计算识别对齐损失 (使用别人的多层次方法)
        # 全局损失
        motion_feat_recog = self.recog_proj_head(j_t_s)
        loss_recog_global = self.loss_fn(motion_feat_recog, text_feat_recog)
        # 局部损失
        t_g_proj_recog = self.recog_proj_head(t_g)
        s_g_proj_recog = self.recog_proj_head(s_g)
        loss_recog_local = (self.loss_fn(t_g_proj_recog, text_feat_recog) + self.loss_fn(s_g_proj_recog, text_feat_recog)) * 0.5
        # (这里可以继续添加分段、分块损失，但为了简化，先用这两部分)
        loss_recog = loss_recog_global + loss_recog_local

        
        return loss_desc, loss_recog
    


class MotionCLIP_Hybrid(nn.Module):
    def __init__(self, in_dim, shared_dim=2048, proj_dim=512, text_transformer_layers=8):
        super().__init__()
        
        # === 1. 动作编码器 (来自代码2，最强的部分) ===
        self.motion_feature_extractor = MultiStreamFeatureExtractor(in_dim, hidden_size=shared_dim//2)

        # === 2. 文本编码器塔 ===
        
        # --- 文本塔 A: 冻结的CLIP，用于识别 ---
        clip_model, _ = clip.load('ViT-B/32', device='cpu', jit=False)
        clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        self.frozen_clip_model = clip_model
        
        # --- 文本塔 B: 可微调的Transformer，用于检索 ---
        # (这部分代码来自你的代码1)
        self.trainable_token_embedding = clip_model.token_embedding
        self.trainable_positional_embedding = clip_model.positional_embedding
        # 我们让 token embedding 可微调，这是一个常见的策略
        # self.trainable_token_embedding.requires_grad = True 
        
        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2, activation="gelu")
        self.trainable_text_transformer = nn.TransformerEncoder(
            textTransEncoderLayer, num_layers=text_transformer_layers)
        self.text_ln = nn.LayerNorm(512)
        # 最后的投影层，将文本特征映射到proj_dim
        self.trainable_text_proj = nn.Linear(512, proj_dim)

        # === 3. 动作双投影头 (我们的思想) ===
        self.desc_proj_head = nn.Sequential(nn.LayerNorm(proj_dim), nn.Linear(proj_dim, proj_dim))
        self.recog_proj_head = nn.Sequential(nn.LayerNorm(proj_dim), nn.Linear(proj_dim, proj_dim))

        # === 4. 损失与温度系数 ===
        self.loss_fn = Contrastive_loss_3()
        self.logit_scale_desc = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_recog = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_text_recognition(self, category_texts):
        """使用冻结的CLIP编码类别文本"""
        device = next(self.parameters()).device
        with torch.no_grad():
            text_tokens = clip.tokenize(category_texts, truncate=True).to(device)
            text_features = self.frozen_clip_model.encode_text(text_tokens).float()
        return text_features

    def encode_text_description(self, desc_texts):
        """使用可训练的Transformer编码描述文本"""
        device = next(self.parameters()).device
        text_tokens = clip.tokenize(desc_texts, truncate=True).to(device)
        
        # 使用可训练的 embedding
        x = self.trainable_token_embedding(text_tokens).float()
        x = x + self.trainable_positional_embedding.float()
        
        x = x.permute(1, 0, 2)
        x = self.trainable_text_transformer(x)
        x = x.permute(1, 0, 2)
        x = self.text_ln(x)
        
        # 取 [EOS] token 的特征
        x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)]
        x = self.trainable_text_proj(x)
        return x

    def forward_loss(self, motion, m_lens, desc_texts, category_texts, lambda_recog=1.0):
        # 1. 提取多层次动作特征
        j_t_s, t_g, s_g, _, _ = self.motion_feature_extractor(motion, m_lens)

        # --- 检索任务通道 ---
        motion_feat_desc = self.desc_proj_head(j_t_s)
        text_feat_desc = self.encode_text_description(desc_texts)
        
        # 计算描述对齐损失
        loss_desc_global = self.loss_fn(motion_feat_desc, text_feat_desc)
        loss_desc_local = (self.loss_fn(t_g, text_feat_desc) + self.loss_fn(s_g, text_feat_desc)) * 0.5
        loss_desc = loss_desc_global + loss_desc_local

        # --- 识别任务通道 ---
        motion_feat_recog = self.recog_proj_head(j_t_s)
        text_feat_recog = self.encode_text_recognition(category_texts)
        
        # 计算识别对齐损失
        loss_recog_global = self.loss_fn(motion_feat_recog, text_feat_recog)
        loss_recog_local = (self.loss_fn(t_g, text_feat_recog) + self.loss_fn(s_g, text_feat_recog)) * 0.5
        loss_recog = loss_recog_global + loss_recog_local

        return loss_desc, loss_recog

    # --- 用于推理的辅助函数 ---
    def encode_motion_description(self, motion, m_lens):
        j_t_s, _, _, _, _ = self.motion_feature_extractor(motion, m_lens)
        return self.desc_proj_head(j_t_s)
        
    def encode_motion_recognition(self, motion, m_lens):
        j_t_s, _, _, _, _ = self.motion_feature_extractor(motion, m_lens)
        return self.recog_proj_head(j_t_s)
    
class MotionCLIP_Hybrid_oneLoss(nn.Module):
    def __init__(self, in_dim, shared_dim=2048, proj_dim=512, text_transformer_layers=8):
        super().__init__()
        
        # === 1. 动作编码器 (来自代码2，最强的部分) ===
        self.motion_feature_extractor = MultiStreamFeatureExtractor(in_dim, hidden_size=shared_dim//2)

        # === 2. 文本编码器塔 ===
        
        # --- 文本塔 A: 冻结的CLIP，用于识别 ---
        clip_model, _ = clip.load('ViT-B/32', device='cpu', jit=False)
        clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        self.clip_model = clip_model
        
        # --- 文本塔 B: 可微调的Transformer，用于检索 ---
        # (这部分代码来自你的代码1)
        self.trainable_token_embedding = clip_model.token_embedding
        self.trainable_positional_embedding = clip_model.positional_embedding
        # 我们让 token embedding 可微调，这是一个常见的策略
        # self.trainable_token_embedding.requires_grad = True 
        
        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2, activation="gelu")
        self.trainable_text_transformer = nn.TransformerEncoder(
            textTransEncoderLayer, num_layers=text_transformer_layers)
        self.text_ln = nn.LayerNorm(512)
        # 最后的投影层，将文本特征映射到proj_dim
        self.trainable_text_proj = nn.Linear(512, proj_dim)

        # === 3. 动作双投影头 (我们的思想) ===
        self.desc_proj_head = nn.Sequential(nn.LayerNorm(proj_dim), nn.Linear(proj_dim, proj_dim))
        self.recog_proj_head = nn.Sequential(nn.LayerNorm(proj_dim), nn.Linear(proj_dim, proj_dim))

        # === 4. 损失与温度系数 ===
        self.loss_fn = Contrastive_loss_3()
        self.logit_scale_desc = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_recog = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_text_recognition(self, category_texts):
        """使用冻结的CLIP编码类别文本"""
        device = next(self.parameters()).device
        with torch.no_grad():
            text_tokens = clip.tokenize(category_texts, truncate=True).to(device)
            text_features = self.clip_model.encode_text(text_tokens).float()
        return text_features

    def encode_text_description(self, desc_texts):
        """使用可训练的Transformer编码描述文本"""
        device = next(self.parameters()).device
        text_tokens = clip.tokenize(desc_texts, truncate=True).to(device)
        
        # 使用可训练的 embedding
        x = self.trainable_token_embedding(text_tokens).float()
        x = x + self.trainable_positional_embedding.float()
        
        x = x.permute(1, 0, 2)
        x = self.trainable_text_transformer(x)
        x = x.permute(1, 0, 2)
        x = self.text_ln(x)
        
        # 取 [EOS] token 的特征
        x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)]
        x = self.trainable_text_proj(x)
        return x

    def forward_loss(self, motion, m_lens, desc_texts, category_texts, lambda_recog=1.0):
        # 1. 提取多层次动作特征
        j_t_s, t_g, s_g, _, _ = self.motion_feature_extractor(motion, m_lens)

        # --- 检索任务通道 ---
        motion_feat_desc = self.desc_proj_head(j_t_s)
        text_feat_desc = self.encode_text_description(desc_texts)
        
        # 计算描述对齐损失
        loss_desc_global = self.loss_fn(motion_feat_desc, text_feat_desc)
        # loss_desc_local = (self.loss_fn(t_g, text_feat_desc) + self.loss_fn(s_g, text_feat_desc)) * 0.5
        loss_desc = loss_desc_global
        # --- 识别任务通道 ---
        motion_feat_recog = self.recog_proj_head(j_t_s)
        text_feat_recog = self.encode_text_recognition(category_texts)
        
        # 计算识别对齐损失
        loss_recog_global = self.loss_fn(motion_feat_recog, text_feat_recog)
        # loss_recog_local = (self.loss_fn(t_g, text_feat_recog) + self.loss_fn(s_g, text_feat_recog)) * 0.5
        loss_recog = loss_recog_global

        return loss_desc, loss_recog

    # --- 用于推理的辅助函数 ---
    def encode_motion_description(self, motion, m_lens):
        j_t_s, _, _, _, _ = self.motion_feature_extractor(motion, m_lens)
        return self.desc_proj_head(j_t_s)
        
    def encode_motion_recognition(self, motion, m_lens):
        j_t_s, _, _, _, _ = self.motion_feature_extractor(motion, m_lens)
        return self.recog_proj_head(j_t_s)
    

class MotionCLIP_Reward_2text(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.motion_encoder = MotionEncoder_Reward(in_dim, 512, 1024, 4, 2, 0.2, 'gelu')

        # --- 文本塔 A: 冻结的CLIP，用于识别 ---
        clip_model, _ = clip.load('ViT-B/32', device='cpu', jit=False)
        clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        self.clip_model = clip_model

        # --- 文本塔 B: 可微调的Transformer，用于检索 ---
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        no_grad(self.token_embedding)

        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=512, nhead=2, dim_feedforward=1024, dropout=0.2, activation="gelu",)
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer, num_layers=4)
        self.text_ln = nn.LayerNorm(512)
        self.out = nn.Linear(512, 512)

        # === 损失与温度系数 ===
        self.loss_fn_recognition = Contrastive_loss_3()
        self.loss_fn_retrieve = InfoNCE_with_filtering(temperature=0.1, threshold_selfsim=0.9)
        # 新增 L1 损失用于 latent_loss
        self.loss_fn_latent = nn.SmoothL1Loss(reduction="mean")



    # *** 修改 encode_motion 以传递 timestep ***
    def encode_motion(self, motion, m_lens, timestep=None):
        seq_len = motion.shape[1]
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        motion_embedding = self.motion_encoder(motion, padding_mask.to(motion.device), timestep)
        return motion_embedding

    def encode_text(self, text):
        # ... (代码不变) ...
        device = next(self.parameters()).device
        with torch.no_grad():
            text_tokens = clip.tokenize(text, truncate=True).to(device)
            x = self.token_embedding(text_tokens).float()
            pe_tokens = x + self.positional_embedding.float()
        pe_tokens = pe_tokens.permute(1,0,2)
        out = self.textTransEncoder(pe_tokens)
        out = out.permute(1, 0, 2)
        out = self.text_ln(out)
        out = out[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)]
        out = self.out(out)
        return out
    
    def encode_text_recognition(self, category_texts):
        """使用冻结的CLIP编码类别文本"""
        device = next(self.parameters()).device
        with torch.no_grad():
            text_tokens = clip.tokenize(category_texts, truncate=True).to(device)
            text_features = self.clip_model.encode_text(text_tokens).float()
        return text_features

    def forward_loss(self, motion, m_lens, desc_texts, category_texts):
        # 1. 提取多层次动作特征
        motion_embedding = self.encode_motion(motion, m_lens)

        text_feat_desc = self.encode_text(desc_texts)
        text_feat_desc_clip = self.encode_text_recognition(desc_texts)
        text_feat_recog = self.encode_text_recognition(category_texts)

        loss_latent = self.loss_fn_latent(motion_embedding, text_feat_desc)
        loss_desc_global = self.loss_fn_retrieve(motion_embedding, text_feat_desc, text_feat_desc_clip)
        loss_recog_global = self.loss_fn_recognition(motion_embedding, text_feat_recog)+0.5*loss_latent

        return loss_desc_global, loss_recog_global
    
    def forward(self, motion, m_lens, desc_texts):
        """
        这个函数现在是训练的核心。
        它使用多流编码器计算复杂的层级化特征。
        """
        # 1. 提取多层次动作特征
        motion_embedding = self.encode_motion(motion, m_lens)

        # 2. 编码两种文本
        text_feat_desc = self.encode_text(desc_texts)

        return motion_embedding, motion_embedding, text_feat_desc, None
    
# 放在模型文件 (e.g., evaluators_action.py) 中
####################################################################################
######下面是多模态后融合的MotionCLIP模型#########
class MultiModalMotionEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim=512, ff_size=1024, num_layers=8, num_heads=8, dropout=0.2, activation='gelu'):
        super().__init__()
        
        # 为每种模态创建一个独立的编码器实例
        self.joint_encoder = MotionEncoder_Reward(in_dim, latent_dim, ff_size, num_layers, num_heads, dropout, activation)
        self.bone_encoder = MotionEncoder_Reward(in_dim, latent_dim, ff_size, num_layers, num_heads, dropout, activation)
        self.motion_encoder_stream = MotionEncoder_Reward(in_dim, latent_dim, ff_size, num_layers, num_heads, dropout, activation)
        
        # 骨骼连接
        self.bones = [(1, 4), (2, 1), (3, 1), (4, 7), (5, 2), (6, 3), (7, 10), (8, 5), (9, 6),
                     (10, 10), (11, 8), (12, 9), (13, 10), (14, 10), (15, 10), (16, 13), (17, 14),
                     (18, 15), (19, 17), (20, 18), (21, 19), (22, 20)]
        
        # 融合层 (一个简单的线性层或更复杂的注意力机制)
        # 3 * latent_dim -> latent_dim
        self.fusion_layer = nn.Sequential(
            nn.LayerNorm(3 * 512),
            nn.Linear(3 * 512, 512)
        )
    def modality_generation(self, motion_joints, bones):
        """
        从关节数据生成骨骼和运动数据。
        :param motion_joints: 关节数据, shape [B, T, D_pose] where D_pose = J*3
        :param bones: 骨骼连接列表
        :return: (motion_bones, motion_motion)
        """
        B, T, D_pose = motion_joints.shape
        J = D_pose // 3
        
        # Reshape to [B, T, J, 3] for easier manipulation
        joints = motion_joints.view(B, T, J, 3)
        
        # --- 1. 计算 Bone 数据 ---
        # Bone = 子关节 - 父关节
        motion_bones = torch.zeros_like(joints)
        for child, parent in bones:
            motion_bones[:, :, child-1, :] = joints[:, :, child-1, :] - joints[:, :, parent-1, :]
        motion_bones = motion_bones.view(B, T, D_pose)
        
        # --- 2. 计算 Motion 数据 (速度) ---
        # Motion = 当前帧 - 上一帧
        motion_motion = torch.zeros_like(joints)
        motion_motion[:, 1:, :, :] = joints[:, 1:, :, :] - joints[:, :-1, :, :]
        motion_motion = motion_motion.view(B, T, D_pose)
        
        return motion_bones, motion_motion
    
    def forward(self, motion, padding_mask, timestep=None):
        # 1. 生成多模态数据
        # 输入 motion 是关节数据
        motion_bones, motion_motion = self.modality_generation(motion, self.bones)

        feat_joint = self.joint_encoder(motion, padding_mask, timestep)
        feat_bone = self.bone_encoder(motion_bones, padding_mask, timestep)
        feat_motion = self.motion_encoder_stream(motion_motion, padding_mask, timestep)
        
        # 3. 特征融合
        # 简单拼接
        combined_features = torch.cat([feat_joint, feat_bone, feat_motion], dim=1)
        
        # 通过融合层
        fused_features = self.fusion_layer(combined_features)
        
        return fused_features, feat_joint, feat_bone, feat_motion
    

class MotionCLIP_Reward_2text_3stream(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        d_model = 512
        layers = 8
        heads = 8
        self.motion_encoder = MultiModalMotionEncoder(in_dim, d_model, 1024, layers, heads, 0.2, 'gelu')

        # --- 文本塔 A: 冻结的CLIP，用于识别 ---
        clip_model, _ = clip.load('ViT-B/32', device='cpu', jit=False)
        clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        self.clip_model = clip_model

        # --- 文本塔 B: 可微调的Transformer，用于检索 ---
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        no_grad(self.token_embedding)

        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=512, nhead=heads, dim_feedforward=1024, dropout=0.2, activation="gelu",)
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer, num_layers=layers)
        self.text_ln = nn.LayerNorm(512)
        self.out = nn.Linear(512, 512)

        # === 损失与温度系数 ===
        self.loss_fn_recognition = Contrastive_loss_3()
        self.loss_fn_retrieve = InfoNCE_with_filtering(temperature=0.1, threshold_selfsim=0.9)
        # 新增 L1 损失用于 latent_loss
        self.loss_fn_latent = nn.SmoothL1Loss(reduction="mean")

    # *** 修改 encode_motion 以传递 timestep ***
    def encode_motion(self, motion, m_lens, timestep=None):
        seq_len = motion.shape[1]
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        motion_embedding, _, _, _ = self.motion_encoder(motion, padding_mask.to(motion.device), timestep)
        return motion_embedding
    
    def encode_3stream(self, motion, m_lens, timestep=None):
        seq_len = motion.shape[1]
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        motion_embedding, feat_joint, feat_bone, feat_motion = self.motion_encoder(motion, padding_mask.to(motion.device))
        return motion_embedding, feat_joint, feat_bone, feat_motion

    def encode_text(self, text):
        # ... (代码不变) ...
        device = next(self.parameters()).device
        with torch.no_grad():
            text_tokens = clip.tokenize(text, truncate=True).to(device)
            x = self.token_embedding(text_tokens).float()
            pe_tokens = x + self.positional_embedding.float()
        pe_tokens = pe_tokens.permute(1,0,2)
        out = self.textTransEncoder(pe_tokens)
        out = out.permute(1, 0, 2)
        out = self.text_ln(out)
        out = out[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)]
        out = self.out(out)
        return out
    
    def encode_text_recognition(self, category_texts):
        """使用冻结的CLIP编码类别文本"""
        device = next(self.parameters()).device
        with torch.no_grad():
            text_tokens = clip.tokenize(category_texts, truncate=True).to(device)
            text_features = self.clip_model.encode_text(text_tokens).float()
        return text_features

    # def forward_loss(self, motion, m_lens, desc_texts, category_texts):
    #     # 1. 提取多层次动作特征
    #     seq_len = motion.shape[1]
    #     padding_mask = ~lengths_to_mask(m_lens, seq_len)
    #     motion_embedding, feat_joint, feat_bone, feat_motion = self.motion_encoder(motion, padding_mask.to(motion.device))
    #     # 2. 编码文本
    #     text_feat_desc = self.encode_text(desc_texts)
    #     text_feat_desc_clip = self.encode_text_recognition(desc_texts)
    #     text_feat_recog = self.encode_text_recognition(category_texts)
    #     # 3. 检索任务通道
    #     loss_latent = self.loss_fn_latent(motion_embedding, text_feat_desc)

    #     loss_desc_global = self.loss_fn_retrieve(motion_embedding, text_feat_desc, text_feat_desc_clip)+loss_latent
    #     # 4. 识别任务通道
    #     loss_recog_global = self.loss_fn_recognition(motion_embedding, text_feat_recog)

    #     return loss_desc_global, loss_recog_global

    def forward_loss(self, motion, m_lens, desc_texts, category_texts):
        # 1. 提取多层次动作特征
        seq_len = motion.shape[1]
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        motion_embedding, feat_joint, feat_bone, feat_motion = self.motion_encoder(motion, padding_mask.to(motion.device))
        # 2. 编码文本
        # text_feat_desc = self.encode_text(desc_texts)
        # text_feat_desc_clip = self.encode_text_recognition(desc_texts)
        text_feat_recog = self.encode_text_recognition(category_texts)
        # 3. 检索任务通道
        # loss_latent = (self.loss_fn_latent(motion_embedding, text_feat_desc)+
        #                self.loss_fn_latent(feat_joint, text_feat_desc)+
        #                self.loss_fn_latent(feat_bone, text_feat_desc)+
        #                self.loss_fn_latent(feat_motion, text_feat_desc))/4.0
        # loss_local = (self.loss_fn_retrieve(feat_joint, text_feat_desc, text_feat_desc_clip)+
        #               self.loss_fn_retrieve(feat_bone, text_feat_desc, text_feat_desc_clip)+
        #               self.loss_fn_retrieve(feat_motion, text_feat_desc, text_feat_desc_clip))/3.0
        # loss_desc_global = self.loss_fn_retrieve(motion_embedding, text_feat_desc, text_feat_desc_clip)+loss_latent+loss_local
        # 4. 识别任务通道
        # loss_recog_global = self.loss_fn_recognition(motion_embedding, text_feat_recog)
        ## 只用识别任务
        # loss_local = (self.loss_fn_recognition(feat_joint, text_feat_recog)+
        #               self.loss_fn_recognition(feat_bone, text_feat_recog)+
        #               self.loss_fn_recognition(feat_motion, text_feat_recog))/3.0
        loss_recog_global = self.loss_fn_recognition(motion_embedding, text_feat_recog)
        # loss_recog_global = 0

        return 0, loss_recog_global
    
    # def forward_loss(self, motion, m_lens, desc_texts, category_texts):
    #     # 1. 提取多层次动作特征
    #     seq_len = motion.shape[1]
    #     padding_mask = ~lengths_to_mask(m_lens, seq_len)
    #     motion_embedding, feat_joint, feat_bone, feat_motion = self.motion_encoder(motion, padding_mask.to(motion.device))
    #     # 2. 编码文本
    #     text_feat_desc = self.encode_text(desc_texts)
    #     # text_feat_desc_clip = self.encode_text_recognition(desc_texts)
    #     text_feat_recog = self.encode_text_recognition(category_texts)
    #     # 3. 检索任务通道
    #     loss_latent = self.loss_fn_latent(motion_embedding, text_feat_desc)

    #     loss_desc_global = self.loss_fn_recognition(motion_embedding, text_feat_desc)+loss_latent
    #     # 4. 识别任务通道
    #     loss_recog_global = self.loss_fn_recognition(motion_embedding, text_feat_recog)

    #     return loss_desc_global, loss_recog_global
    
    def forward(self, motion, m_lens, desc_texts):
        """
        这个函数现在是训练的核心。
        它使用多流编码器计算复杂的层级化特征。
        """
        # 1. 提取多层次动作特征
        motion_embedding = self.encode_motion(motion, m_lens)

        # 2. 编码两种文本
        text_feat_desc = self.encode_text(desc_texts)

        return motion_embedding, motion_embedding, text_feat_desc, None

    def retrieve_closest_motion(self, text_features, database):
        """
        根据给定的文本特征，在数据库中检索最相似的真实动作。
        :param text_features: 编码后的文本特征, shape [B, D]
        :param database: 预计算好的动作数据库
        :return: (retrieved_motions, retrieved_mlens)
        """
        # 确保数据库特征在正确的设备上
        db_features = database['features'].to(text_features.device)
        
        # 对查询文本特征进行归一化
        text_features_norm = F.normalize(text_features, dim=1)
        
        # 计算相似度 (矩阵乘法，因为特征都已归一化)
        # sim_matrix.shape: [B_query, N_database]
        sim_matrix = text_features_norm @ db_features.t()
        
        # 找到每个查询最匹配的索引
        # best_indices.shape: [B_query]
        _, best_indices = torch.topk(sim_matrix, k=1, dim=1)
        best_indices = best_indices.squeeze(1).cpu()
        
        # 根据索引从数据库中取出对应的真实动作和长度
        retrieved_motions = database['motions'][best_indices]
        retrieved_mlens = database['mlens'][best_indices]
        
        return retrieved_motions.to(text_features.device), retrieved_mlens.to(text_features.device)
    
# 放在模型文件 (e.g., evaluators_action.py) 中


##########################################################################################################
#####   下面是新的 MultiModalMotionEncoder_EarlyFusion 相关代码   #####
# 这是一个新的、专门用于将动作序列转换为嵌入序列的模块
class MotionEmbedding(nn.Module):
    def __init__(self, in_dim, latent_dim, dropout=0.2):
        super().__init__()
        self.embed_motion = nn.Linear(in_dim, latent_dim)
        self.sequence_pos_encoder = PositionalEncodingCLIP(latent_dim, dropout, max_len=2000)

    def forward(self, motion):
        # motion: [B, T, D]
        x_emb = self.embed_motion(motion)
        x_pos_emb = self.sequence_pos_encoder(x_emb)
        return x_pos_emb

# 主编码器，现在只包含Transformer和输出部分
class MotionTransformer(nn.Module):
    def __init__(self, latent_dim=512, ff_size=1024, num_layers=8, num_heads=8, dropout=0.2, activation='gelu'):
        super().__init__()
        self.latent_dim = latent_dim
        self.query_token = nn.Parameter(torch.randn(1, self.latent_dim))
        self.time_embed = TimestepEmbedder(self.latent_dim) # Timestep 嵌入器

        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim, nhead=num_heads,
            dim_feedforward=ff_size, dropout=dropout, activation=activation)
        self.transformer = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=num_layers)
        
        self.out_ln = nn.LayerNorm(self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 512) # 输出维度固定为512

    def forward(self, fused_embedding, padding_mask, timestep=None):
        # fused_embedding: [B, T, latent_dim]
        B, T, D = fused_embedding.shape
        
        # # 添加 timestep 嵌入
        # time_emb = self.time_embed(timestep) if timestep is not None else 0
        # x_with_time = fused_embedding + time_emb.unsqueeze(1)
        
        # 拼接 query_token
        emb = torch.cat([self.query_token.repeat(B, 1, 1), fused_embedding], dim=1)
        
        # 创建 query_token 对应的 mask
        query_mask = torch.zeros(B, 1, dtype=torch.bool, device=emb.device)
        padding_mask = torch.cat([query_mask, padding_mask], dim=1)
        
        h = emb.permute(1, 0, 2)
        h = self.transformer(h, src_key_padding_mask=padding_mask)
        h = h.permute(1, 0, 2)
        h = self.out_ln(h)
        
        # 只取 query_token 的输出
        motion_emb = self.out(h[:, 0])
        return motion_emb
    
# 重构后的 MultiModalMotionEncoder

class MultiModalMotionEncoder_EarlyFusion(nn.Module):
    def __init__(self, in_dim, latent_dim=512, ff_size=1024, num_layers=8, num_heads=8, dropout=0.2, activation='gelu'):
        super().__init__()
        
        # --- 1. 浅层嵌入层 (Early Stage) ---
        # 为每种模态创建一个独立的嵌入层
        self.joint_embedding = MotionEmbedding(in_dim, latent_dim, dropout)
        self.bone_embedding = MotionEmbedding(in_dim, latent_dim, dropout)
        self.motion_embedding = MotionEmbedding(in_dim, latent_dim, dropout)
        
        # --- 2. 深度主干编码器 (Shared Backbone) ---
        # 只有一个 Transformer 主干
        self.transformer_backbone = MotionTransformer(latent_dim, ff_size, num_layers, num_heads, dropout, activation)
        
        # 骨骼连接 (保持不变)
        self.bones = [(1, 4), (2, 1), (3, 1), (4, 7), (5, 2), (6, 3), (7, 10), (8, 5), (9, 6),
                     (10, 10), (11, 8), (12, 9), (13, 10), (14, 10), (15, 10), (16, 13), (17, 14),
                     (18, 15), (19, 17), (20, 18), (21, 19), (22, 20)]
        
        # (不再需要后期融合的 fusion_layer)

    def modality_generation(self, motion_joints, bones):
        """
        从关节数据生成骨骼和运动数据。
        :param motion_joints: 关节数据, shape [B, T, D_pose] where D_pose = J*3
        :param bones: 骨骼连接列表
        :return: (motion_bones, motion_motion)
        """
        B, T, D_pose = motion_joints.shape
        J = D_pose // 3
        
        # Reshape to [B, T, J, 3] for easier manipulation
        joints = motion_joints.view(B, T, J, 3)
        
        # --- 1. 计算 Bone 数据 ---
        # Bone = 子关节 - 父关节
        motion_bones = torch.zeros_like(joints)
        for child, parent in bones:
            motion_bones[:, :, child-1, :] = joints[:, :, child-1, :] - joints[:, :, parent-1, :]
        motion_bones = motion_bones.view(B, T, D_pose)
        
        # --- 2. 计算 Motion 数据 (速度) ---
        # Motion = 当前帧 - 上一帧
        motion_motion = torch.zeros_like(joints)
        motion_motion[:, 1:, :, :] = joints[:, 1:, :, :] - joints[:, :-1, :, :]
        motion_motion = motion_motion.view(B, T, D_pose)
        
        return motion_bones, motion_motion
    
    def forward(self, motion, padding_mask, timestep=None):
        # 1. 生成多模态数据
        motion_bones, motion_motion = self.modality_generation(motion, self.bones)

        # 2. 并行进行浅层嵌入
        emb_joint = self.joint_embedding(motion)
        emb_bone = self.bone_embedding(motion_bones)
        emb_motion = self.motion_embedding(motion_motion)
        
        # 3. 早期融合 (Early Fusion)
        # 通过平均或相加来融合
        fused_embedding = (emb_joint + emb_bone + emb_motion) / 3.0

        final_features = self.transformer_backbone(fused_embedding, padding_mask, timestep)
        
        return final_features