import numpy as np
import torch

from coamd.evaluation.metrics import calculate_R_precision


def cosine_similarity_matrix(a, b):
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)
    a = a.float()
    b = b.float()
    a = a / (a.norm(dim=1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=1, keepdim=True) + 1e-8)
    return a @ b.t()
