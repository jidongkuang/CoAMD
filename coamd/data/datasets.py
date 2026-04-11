# 将以下代码添加到 utils/datasets.py 的末尾
from os.path import join as pjoin
from torch.utils import data
import numpy as np
import random
import orjson
import codecs as cs
from coamd.utils.glove import GloVe
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate

#################################################################################
#                                  Collate Function                             #
#################################################################################
def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

#################################################################################
#                                      Datasets                                 #
#################################################################################
class ActionDataset(data.Dataset):
    """
    用于动作识别任务的数据集。
    在__getitem__中返回 (动作片段, 对应的文本特征)。
    """
    def __init__(self, mean, std, split_file, dataset_name, motion_dir, text_dir, window_size=64, normalization=True):
        
        self.max_length = 20
        self.pointer = 0
        self.window_size = window_size
        self.dataset_name = dataset_name
        # self.max_text_len = max_text_len
        # self.unit_length = unit_length
        self.normalization = normalization

        annotations_actions_path = pjoin(motion_dir, '..', 'annotations_actions_400.json')
        with open(annotations_actions_path, "rb") as ff:
            self.annotations_actions = orjson.loads(ff.read())
        #     self.joints_num = 22
        min_motion_len = 40 if self.dataset_name =='t2m' else 24

        # joints_num = self.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(motion_dir, name + '.npy'))
                motion = motion.reshape(motion.shape[0], -1)  # (t, 22*3)
                if (len(motion)) < self.window_size or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + '.txt')) as f:
                    action = self.annotations_actions[name]
                    for line_number, line in enumerate(f.readlines()):
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = action['annotations'][line_number]['processed_label_text']
                        text_dict['label'] = action['annotations'][line_number]['label']
                        text_dict['label_list'] = action['labels']
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < self.window_size or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                # print(e)
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return (data - self.mean) / self.std
        else:
            return (data - mean) / std

    def inv_transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return data * self.std + self.mean
        else:
            return data * std + mean
        
    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        label = text_data['label']
        label = random.choice(label)

        label_list = text_data['label_list']
        # 生成multi-hot编码
        # 创建一个全零的numpy数组
        multi_hot = np.zeros(400, dtype=np.float32)
        # 使用高级索引一次性设置多个位置为1.0
        multi_hot[label_list] = 1.0

        # if self.unit_length < 10:
        #     coin2 = np.random.choice(['single', 'single', 'double'])
        # else:
        #     coin2 = 'single'

        # if coin2 == 'double':
        #     m_length = (m_length // self.unit_length - 1) * self.unit_length
        # elif coin2 == 'single':
        #     m_length = (m_length // self.unit_length) * self.unit_length
        # idx = random.randint(0, len(motion) - m_length)
        # motion = motion[idx:idx+m_length]  ## (t, 22, 3)

        "Z Normalization"
        if self.normalization:
            motion = (motion - self.mean) / self.std  ## (t,22,3)

        idx = random.randint(0, len(motion) - self.window_size)  # 包含两端点
        motion = motion[idx:idx + self.window_size]  ## (64, 22, 3)
        

        # data_numpy = np.expand_dims(motion, axis=0)  ## (1,t,22,3)
        # data_numpy = np.pad(data_numpy, ((0, 1), (0, 0), (0, 11), (0, 0)), mode='constant')
        # data_numpy = data_numpy[:,:,self.joint_sequence,:]
        # # data: m t v c -> c t v m
        # data_numpy = data_numpy.transpose(3, 1, 2, 0)

        # # label_text = clip.tokenize(caption, truncate=True).numpy().squeeze()
        # label_text = self.label_texts[label]

        # valid_frame_num = m_length
        # data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
            
        return motion, label, multi_hot
    

class AE_ActionDataset(data.Dataset):
    """
    用于动作识别任务的数据集，采用 cumsum 机制来索引所有可能的动作片段。
    """
    def __init__(self, mean, std, split_file, motion_dir, annotations_actions, window_size=64):
        """
        初始化数据集。
        :param mean: 动作数据均值
        :param std: 动作数据标准差
        :param motion_dir: 动作 .npy 文件目录
        :param text_dir: 文本 .txt 文件目录
        :param split_file: 数据划分文件，每行一个文件名（不含后缀）
        :param window_size: 训练时采样的窗口大小
        :param annotations_actions: 从JSON加载的动作标注信息
        """
        self.mean = mean
        self.std = std
        self.window_size = window_size
        with open(annotations_actions, "rb") as ff:
            self.annotations_actions = orjson.loads(ff.read())

        self.motions = []  # 存储加载的动作数据 (np.array)
        self.labels = []   # 存储与每个动作文件对应的标签列表 (list of ints)
        self.annotations = [] 
        self.lengths = []  # 存储每个动作可以采样的片段数

        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        print(f"Loading data from {len(id_list)} files...")
        for name in tqdm(id_list):
            try:
                # --- 加载动作数据 ---
                motion_path = pjoin(motion_dir, name + '.npy')
                motion = np.load(motion_path)

                # --- 过滤短动作 ---
                if motion.shape[0] < self.window_size:
                    continue

                # --- 加载对应的标签 ---
                # 假设每个动作文件对应一个或多个文本，但共享相同的动作类别标签
                # 我们只取第一个文本描述对应的标签作为整个动作的标签
                # text_path = pjoin(text_dir, name + '.txt')
                action_info = self.annotations_actions[name]
                label_list = action_info['labels']
                annotations_list = action_info['annotations']
                
                # if not action_info or not action_info.get('labels'):
                #     # 如果在标注文件中找不到该动作或没有标签，则跳过
                #     continue
                
                # 获取与整个动作文件关联的标签列表 (不是某个片段)
                # 官方代码中，f_tag=0, to_tag=0 表示整个文件
                # 我们这里简化，直接使用文件根级别的 'labels'

                # --- 存储数据 ---
                self.motions.append(motion)
                self.labels.append(label_list)
                self.annotations.append(annotations_list)
                # 计算这个动作可以产生多少个 window_size 的片段
                self.lengths.append(motion.shape[0] - self.window_size + 1)

            except Exception as e:
                # print(f"Skipping file {name} due to error: {e}")
                pass
        
        # --- 创建累积和索引 ---
        if not self.lengths:
            raise ValueError("No valid data found. Check paths and data integrity.")
            
        self.cumsum = np.cumsum([0] + self.lengths)
        
        print(f"Total number of valid motions: {len(self.motions)}")
        print(f"Total number of training snippets: {self.cumsum[-1]}")

    def __len__(self):
        """ 返回总的片段数量 """
        return self.cumsum[-1]

    def __getitem__(self, item):
        """
        根据片段索引 item，返回一个动作片段及其对应的标签。
        """
        # 1. 找到该片段属于哪个动作文件 (motion_id)
        motion_id = np.searchsorted(self.cumsum, item + 1) - 1
        
        # 2. 计算在该动作文件内的起始帧 (start_frame)
        start_frame = item - self.cumsum[motion_id]
        
        # 3. 采样动作片段
        motion_clip = self.motions[motion_id][start_frame : start_frame + self.window_size]
        
        # 4. "Z Normalization"
        # 假设原始数据是 (T, J, 3)，需要先reshape
        original_shape = motion_clip.shape
        motion_clip = motion_clip.reshape(original_shape[0], -1)
        
        # 裁剪到与mean/std匹配的维度
        if motion_clip.shape[1] > self.mean.shape[0]:
             motion_clip = motion_clip[:, :self.mean.shape[0]]

        motion_clip = (motion_clip - self.mean) / self.std
        
        # # 恢复到 (T, J, 3) 形状，如果需要的话
        # motion_clip = motion_clip.reshape(original_shape[0], original_shape[1], original_shape[2])

        # 5. 获取标签
        label_list = self.labels[motion_id]
        annotations = self.annotations[motion_id]
        
        # 从该动作的标签列表中随机选择一个标签
        # (这保留了您原始代码中的逻辑，即一个动作可能对应多个类别)
        label = random.choice(random.choice(annotations)['label'])
        
        # 生成 multi-hot 编码
        multi_hot = np.zeros(400, dtype=np.float32) # 假设总类别数为400
        if label_list:
            multi_hot[label_list] = 1.0

        return motion_clip, label, multi_hot

class AE_ActionDataset_kit(data.Dataset):
    """
    用于动作识别任务的数据集，采用 cumsum 机制来索引所有可能的动作片段。
    """
    def __init__(self, mean, std, split_file, motion_dir, window_size=64):
        """
        初始化数据集。
        :param mean: 动作数据均值
        :param std: 动作数据标准差
        :param motion_dir: 动作 .npy 文件目录
        :param text_dir: 文本 .txt 文件目录
        :param split_file: 数据划分文件，每行一个文件名（不含后缀）
        :param window_size: 训练时采样的窗口大小
        :param annotations_actions: 从JSON加载的动作标注信息
        """
        self.mean = mean
        self.std = std
        self.window_size = window_size

        self.motions = []  # 存储加载的动作数据 (np.array)
        self.labels = []   # 存储与每个动作文件对应的标签列表 (list of ints)
        self.annotations = [] 
        self.lengths = []  # 存储每个动作可以采样的片段数

        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        print(f"Loading data from {len(id_list)} files...")
        for name in tqdm(id_list):
            try:
                # --- 加载动作数据 ---
                motion_path = pjoin(motion_dir, name + '.npy')
                motion = np.load(motion_path)

                # --- 过滤短动作 ---
                if motion.shape[0] < self.window_size:
                    continue

               

                # --- 存储数据 ---
                self.motions.append(motion)

                # 计算这个动作可以产生多少个 window_size 的片段
                self.lengths.append(motion.shape[0] - self.window_size + 1)

            except Exception as e:
                # print(f"Skipping file {name} due to error: {e}")
                pass
        
        # --- 创建累积和索引 ---
        if not self.lengths:
            raise ValueError("No valid data found. Check paths and data integrity.")
            
        self.cumsum = np.cumsum([0] + self.lengths)
        
        print(f"Total number of valid motions: {len(self.motions)}")
        print(f"Total number of training snippets: {self.cumsum[-1]}")

    def __len__(self):
        """ 返回总的片段数量 """
        return self.cumsum[-1]

    def __getitem__(self, item):
        """
        根据片段索引 item，返回一个动作片段及其对应的标签。
        """
        # 1. 找到该片段属于哪个动作文件 (motion_id)
        motion_id = np.searchsorted(self.cumsum, item + 1) - 1
        
        # 2. 计算在该动作文件内的起始帧 (start_frame)
        start_frame = item - self.cumsum[motion_id]
        
        # 3. 采样动作片段
        motion_clip = self.motions[motion_id][start_frame : start_frame + self.window_size]
        
        # 4. "Z Normalization"
        # 假设原始数据是 (T, J, 3)，需要先reshape
        original_shape = motion_clip.shape
        motion_clip = motion_clip.reshape(original_shape[0], -1)
        
        # 裁剪到与mean/std匹配的维度
        if motion_clip.shape[1] > self.mean.shape[0]:
             motion_clip = motion_clip[:, :self.mean.shape[0]]

        motion_clip = (motion_clip - self.mean) / self.std
        
        # # 恢复到 (T, J, 3) 形状，如果需要的话
        # motion_clip = motion_clip.reshape(original_shape[0], original_shape[1], original_shape[2])
       

        return motion_clip, 0, 0

class Text2MotionDataset(data.Dataset):
    def __init__(self, mean, std, split_file, dataset_name, motion_dir, text_dir, unit_length, max_motion_length,
                 max_text_length, evaluation=False):
        self.evaluation = evaluation
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.max_text_len = max_text_length
        self.unit_length = unit_length
        min_motion_len = 40 if dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(motion_dir, name + '.npy'))
                motion = motion.reshape(motion.shape[0], -1)  # (t, 22*3)
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass
        if self.evaluation:
            self.w_vectorizer = GloVe('./glove', 'our_vab')
            name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        else:
            name_list, length_list = new_name_list, length_list
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        if self.evaluation:
            self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return (data - self.mean) / self.std
        else:
            return (data - mean) / std

    def inv_transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return data * self.std + self.mean
        else:
            return data * std + mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if self.evaluation:
            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        # motion = motion[:, :self.mean.shape[0]]
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        elif m_length > self.max_motion_length:
            if not self.evaluation:
                idx = random.randint(0, self.max_motion_length - m_length)
                motion = motion[idx:idx + self.max_motion_length]

        if self.evaluation:
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)
        else:
            return caption, motion, m_length
        
class Text2Motion_Reward_Action_Dataset(data.Dataset):
    def __init__(self, mean, std, split_file, dataset_name, motion_dir, text_dir, annotations_path, label_text_map, unit_length, max_motion_length,
                 max_text_length, evaluation=False):
        self.evaluation = evaluation
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.max_text_len = max_text_length
        self.unit_length = unit_length
        self.label_text_map = label_text_map
        min_motion_len = 40 if dataset_name =='t2m' else 24

        # 加载动作标注文件
        with open(annotations_path, "rb") as f:
            self.annotations_actions = orjson.loads(f.read())

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(motion_dir, name + '.npy'))
                motion = motion.reshape(motion.shape[0], -1)  # (t, 22*3)
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                action_info = self.annotations_actions[name]
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + '.txt')) as f:
                    for line_number, line in enumerate(f.readlines()):
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        text_dict['label'] = action_info['annotations'][line_number]['label']
                        text_dict['label_list'] = action_info['labels']
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass
        if self.evaluation:
            self.w_vectorizer = GloVe('./glove', 'our_vab')
            name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        else:
            name_list, length_list = new_name_list, length_list
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        if self.evaluation:
            self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return (data - self.mean) / self.std
        else:
            return (data - mean) / std

    def inv_transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return data * self.std + self.mean
        else:
            return data * std + mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens, label, label_list = text_data['caption'], text_data['tokens'], text_data['label'], text_data['label_list']
        label = random.choice(label)
        label_text = self.label_text_map[label]
        # 生成 multi-hot 编码
        multi_hot = np.zeros(400, dtype=np.float32) # 假设总类别数为400
        multi_hot[label_list] = 1.0

        if self.evaluation:
            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        # motion = motion[:, :self.mean.shape[0]]
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        elif m_length > self.max_motion_length:
            if not self.evaluation:
                idx = random.randint(0, self.max_motion_length - m_length)
                motion = motion[idx:idx + self.max_motion_length]

        if self.evaluation:
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)
        else:
            return caption, motion, m_length, label, label_text, multi_hot
        