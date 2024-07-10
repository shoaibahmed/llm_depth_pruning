import torch
import numpy as np
from typing import List, Dict

from dist_utils import reduce_tensor


class BlockInfluenceEstimator:
    """
    Implemented from paper: https://arxiv.org/abs/2403.03853
    This influence estimator assumes that the importance of a block is directly related to the size of
        the change it induces to the hidden representation.
    """
    def __init__(self, num_layers: int, device: torch.device, use_avg: bool = True):
        self.num_layers = num_layers
        self.device = device
        self.use_avg = use_avg

        # Initialize the counters
        self.cosine_similarity_dict = {i: 0. for i in range(self.num_layers)}
        self.total_dict = {i: 0 for i in range(self.num_layers)}

    @torch.no_grad()
    def update_block_stats(self, block_idx: int, prev_rep: torch.Tensor, updated_rep: torch.Tensor):
        cosine_sim = torch.nn.functional.cosine_similarity(prev_rep, updated_rep, dim=-1)  # BLD format
        num_elements = np.prod(cosine_sim.shape)  # all others should have the same shape
        cosine_sim = cosine_sim.mean() if self.use_avg else cosine_sim.sum()  # sum cosine similarity over batch and token position
        self.cosine_similarity_dict[block_idx] += float(cosine_sim)
        self.total_dict[block_idx] += 1 if self.use_avg else num_elements

    def get_block_influence(self, block_idx: int) -> float:
        if self.total_dict[block_idx] == 0:  # block not used
            return None
        avg_cosine_sim = self.cosine_similarity_dict[block_idx] / self.total_dict[block_idx]
        avg_cosine_sim = float(reduce_tensor(torch.tensor(avg_cosine_sim).to(self.device), average=True))  # collect from processes
        avg_cosine_dist = 1. - avg_cosine_sim
        return avg_cosine_dist

    def get_block_influences(self) -> List[float]:
        return [self.get_block_influence(i) for i in range(self.num_layers)]

    def __repr__(self) -> str:
        return f"[Block influence estimator]"
