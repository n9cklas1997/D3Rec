import torch
import torch.nn as nn
import math

class DiffRecAdapter(nn.Module):
    """
    Diffusion model adapter.
    Predicts epsilon given (x_t, t).
    """

    def __init__(self, dnn_model, num_steps, t_emb_dim=128):
        super().__init__()

        self.dnn = dnn_model
        self.num_steps = num_steps
        self.t_emb_dim = t_emb_dim

        self.input_dim = dnn_model.dnn.in_dims[0]

        self.time_embedding = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )

        self.t_proj = nn.Linear(t_emb_dim, self.input_dim)

    def sinusoidal_embedding(self, t, dim):
        device = t.device
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32) * -emb
        )
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    def forward(self, x_t, t):
        if t.dim() == 0:
            t = t.unsqueeze(0)

        t_emb = self.sinusoidal_embedding(t, self.t_emb_dim)
        t_emb = self.time_embedding(t_emb)
        t_emb = self.t_proj(t_emb)

        x_in = x_t + t_emb
        return self.dnn(x_in)


class DNNNoTime(nn.Module):
    """
    Wrapper to make DiffRec DNN compatible with diffusion training.
    """
    def __init__(self, dnn):
        super().__init__()
        self.dnn = dnn

    def forward(self, x):
        dummy_t = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
        return self.dnn(x, dummy_t)
