import torch
import torch.nn as nn

#################################################################################
#                                         AE                                    #
#################################################################################
# HumanML3D (SMPL 22 joints) bone structure
t2m_BONES = [(1, 4), (2, 1), (3, 1), (4, 7), (5, 2), (6, 3), (7, 10), (8, 5), (9, 6),
            (10, 10), (11, 8), (12, 9), (13, 10), (14, 10), (15, 10), (16, 13), (17, 14),
            (18, 15), (19, 17), (20, 18), (21, 19), (22, 20)]
kit_BONES = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 4), (6, 4), (7, 6), (8, 7), (9, 4),
    (10, 9), (11, 10), (12, 1), (13, 12), (14, 13), (15, 14), (16, 15), (17, 1),
    (18, 11), (19, 18), (20, 19), (21, 20)]

# --- 2. 新增：多模态编码器 ---
class MultiModalEncoder(nn.Module):
    def __init__(self, input_width, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation='relu', norm=None):
        super().__init__()
        
        # 为每种模态创建一个浅层的输入投影层
        # input_width -> width
        self.joint_proj = nn.Conv1d(input_width, width, kernel_size=3, stride=1, padding=1)
        self.bone_proj = nn.Conv1d(input_width, width, kernel_size=3, stride=1, padding=1)
        self.motion_proj = nn.Conv1d(input_width, width, kernel_size=3, stride=1, padding=1)
        
        # 共享的主干网络 (来自原始 Encoder 的核心部分)
        blocks = []
        blocks.append(nn.ReLU())
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, kernel_size=stride_t * 2, stride=stride_t, padding=stride_t // 2),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, kernel_size=3, stride=1, padding=1))
        self.main_encoder = nn.Sequential(*blocks)

    def forward(self, joint_data, bone_data, motion_data):
        # 输入 shape: [B, T, D_pose]
        # 转换为 Conv1d 需要的 shape: [B, D_pose, T]
        joint_data = joint_data.permute(0, 2, 1)
        bone_data = bone_data.permute(0, 2, 1)
        motion_data = motion_data.permute(0, 2, 1)

        # 1. 独立投影
        proj_joint = self.joint_proj(joint_data)
        proj_bone = self.bone_proj(bone_data)
        proj_motion = self.motion_proj(motion_data)

        # 2. 早期融合 (相加或平均)
        fused_features = (proj_joint + proj_bone + proj_motion) / 3.0

        # 3. 通过共享主干网络进行深度编码
        encoded_latent = self.main_encoder(fused_features)
        
        return encoded_latent # shape: [B, output_emb_width, T_downsampled]

class AE(nn.Module):
    def __init__(self, input_width=67, output_emb_width=512, proj_dim=512, down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None):
        super().__init__()
        self.output_emb_width = output_emb_width
        self.encoder = MultiModalEncoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                                       dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        if input_width == 22*3:
            self.bones = t2m_BONES
        elif input_width == 21*3:
            self.bones = kit_BONES
        # self.projector = nn.Sequential(
        #     nn.Linear(output_emb_width, output_emb_width),
        #     nn.BatchNorm1d(output_emb_width),
        #     nn.ReLU(True),
        #     nn.Linear(output_emb_width, proj_dim),
        # )

    def modality_generation(self, motion_joints, bones):
        B, T, D_pose = motion_joints.shape
        J = D_pose // 3
        joints = motion_joints.view(B, T, J, 3)
        
        # 计算 Bone
        motion_bones = torch.zeros_like(joints)
        for child, parent in bones:
            motion_bones[:, :, child-1, :] = joints[:, :, child-1, :] - joints[:, :, parent-1, :]
        motion_bones = motion_bones.view(B, T, D_pose)
        
        # 计算 Motion (速度)
        motion_motion = torch.zeros_like(joints)
        motion_motion[:, 1:, :, :] = joints[:, 1:, :, :] - joints[:, :-1, :, :]
        motion_motion = motion_motion.view(B, T, D_pose)
        
        return motion_bones, motion_motion

    def preprocess(self, x):
        return x.float()

    def encode(self, x):
        x = self.preprocess(x)
         # 生成 bone 和 motion 数据
        bone_data, motion_data = self.modality_generation(x, self.bones)
        
        # 将三种模态传入编码器
        x_encoder = self.encoder(x, bone_data, motion_data)
        return x_encoder

    def forward(self, x):
        # 编码
        latent_seq = self.encode(x) # shape: [B, D_latent, T_down]
        
        # 解码 (重建)
        x_out = self.decoder(latent_seq) # shape: [B, T, D_pose]
        
        # 投影 (用于识别)
        # 需要对时间维度进行池化来得到全局特征
        # global_latent = latent_seq.mean(dim=2) # [B, D_latent]
        # visual_features = self.projector(global_latent) # [B, D_proj]

        return x_out
    
    def forward_3stream(self, x):
        # 编码
        latent_seq = self.encode(x) # shape: [B, D_latent, T_down]
        
        # 解码 (重建)
        pred_joint = self.decoder(latent_seq) # shape: [B, T, D_pose]
        
        pred_bone, pred_motion = self.modality_generation(pred_joint, self.bones)
        # 投影 (用于识别)
        # 需要对时间维度进行池化来得到全局特征
        global_latent = latent_seq.mean(dim=2) # [B, D_latent]
        # visual_features = self.projector(global_latent) # [B, D_proj]

        return pred_joint, pred_bone, pred_motion

    def decode(self, x):
        x_out = self.decoder(x)
        return x_out

#################################################################################
#                                      AE Zoos                                  #
#################################################################################
def ae(**kwargs):
    return AE(output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None, **kwargs)

AE_models = {
    'AE_Model': ae
}

#################################################################################
#                                 Inner Architectures                           #
#################################################################################
class Encoder(nn.Module):
    def __init__(self, input_emb_width=3, output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, input_emb_width=3, output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None):
        super().__init__()
        blocks = []
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.model(x)
        return x.permute(0, 2, 1)


class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None):
        super().__init__()
        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm)
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class nonlinearity(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=0.2):
        super(ResConv1DBlock, self).__init__()
        padding = dilation
        self.norm = norm

        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()

        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()

        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0, )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = self.dropout(x)
        x = x + x_orig
        return x
