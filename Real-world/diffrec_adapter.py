import torch
import torch.nn as nn
from DiffRecGutter.DiffRec.models.gaussian_diffusion import ModelMeanType

class DiffRecAdapter(nn.Module):
    """
    Makes DiffRec DNN compatible with D3Rec training & sampling.
    Always returns x0_hat.
    """
    def __init__(self, dnn_model, mean_type, d3_diffusion):
        super().__init__()
        self.dnn = dnn_model
        self.mean_type = mean_type
        self.diffusion = d3_diffusion

    def forward(self, x_t, t, **kwargs):
        """
        Args:
            x_t: [B, n_items]
            t:   [B]
        Returns:
            x0_hat: [B, n_items]
        """
        eps_or_x0 = self.dnn(x_t, t)

        if self.mean_type == ModelMeanType.START_X:
            return eps_or_x0

        if self.mean_type == ModelMeanType.EPSILON:
            sqrt_ab = self.diffusion.sqrt_alpha_bar[t].view(-1, 1)
            sqrt_1mab = self.diffusion.sqrt_one_minus_alpha_bar[t].view(-1, 1)
            x0_hat = (x_t - sqrt_1mab * eps_or_x0) / sqrt_ab
            return x0_hat

        raise ValueError(f"Unsupported mean type {self.mean_type}")
