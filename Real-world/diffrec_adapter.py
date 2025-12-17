import torch.nn as nn
from DiffRecGutter.DiffRec.models.gaussian_diffusion import ModelMeanType
from DiffRecGutter.DiffRec.models.DNN import DNN

class DiffRecAdapter(nn.Module):
    def __init__(self, dnn_model, mean_type, d3_diffusion):
        super().__init__()
        self.model = dnn_model
        self.mean_type = mean_type
        self.diffusion = d3_diffusion

    def forward(self, x_t, t, probs=None, probs_mask=None):
        # DiffRec DNN only takes (x_t, t)
        out = self.model(x_t, t)

        if self.mean_type == ModelMeanType.START_X:
            return out

        elif self.mean_type == ModelMeanType.EPSILON:
            # ε → x0 conversion using D3Rec diffusion stats
            sqrt_alpha_bar = self.diffusion.sqrt_alpha_bar[t][:, None]
            sqrt_one_minus_alpha_bar = self.diffusion.sqrt_one_minus_alpha_bar[t][:, None]
            x0_hat = (x_t - sqrt_one_minus_alpha_bar * out) / sqrt_alpha_bar
            return x0_hat

        else:
            raise ValueError("Unknown mean type")
