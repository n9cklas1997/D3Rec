import torch
import torch.nn as nn
from DiffRecGutter.DiffRec.models.gaussian_diffusion import ModelMeanType

class DiffRecAdapter(nn.Module):
    """
    Adapter to make DiffRec DNN compatible with D3Rec's training interface.
    Fixed: produces meaningful x_hat_prob for evaluation.
    """
    def __init__(self, dnn_model, mean_type, d3_diffusion, num_cate):
        super().__init__()
        self.dnn = dnn_model
        self.mean_type = mean_type
        self.diffusion = d3_diffusion
        self.num_cate = num_cate  # number of categories

    def forward(self, x_t, t, prob=None, prob_mask=None, matrix_F=None, recon_only=False):
        """
        Forward pass for D3Rec training interface.

        Args:
            x_t: [batch, num_items] noisy input
            t: timestep (int or tensor of shape [batch])
            prob: optional probability mask
            prob_mask: optional mask for probabilities
            matrix_F: item-category matrix [num_cate, num_items]
            recon_only: if True, return only x_hat_recon (used for sampling)

        Returns:
            If recon_only:
                x_hat_recon: reconstructed x0
            Else:
                x_hat_recon: reconstructed x0
                x_hat_prob: predicted probabilities [batch, num_items]
                x_hat_cate: predicted category embeddings [batch, num_cate, num_items]
                loss_ortho: orthogonal loss (dummy)
        """
        # Ensure t is a tensor
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x_t.device)
        t = t.long()

        # DNN prediction (epsilon or x0)
        eps_or_x0 = self.dnn(x_t, t)  # shape: [batch, num_items]

        # Convert to x0_hat depending on mean type
        if self.mean_type == ModelMeanType.EPSILON:
            # batch-wise indexing of sqrt_alpha_bar and sqrt_one_minus_alpha_bar
            sqrt_ab = self.diffusion.sqrt_alpha_bar[t].unsqueeze(1)         # [batch, 1]
            sqrt_1mab = self.diffusion.sqrt_one_minus_alpha_bar[t].unsqueeze(1)  # [batch, 1]
            x0_hat = (x_t - sqrt_1mab * eps_or_x0) / sqrt_ab
        elif self.mean_type == ModelMeanType.START_X:
            x0_hat = eps_or_x0
        else:
            raise ValueError(f"Unknown mean type {self.mean_type}")

        x_hat_recon = x0_hat

        if recon_only:
            return x_hat_recon  # only reconstruction, used in diffusion sampling

        # --- Meaningful probability prediction ---
        # Softmax over items to produce ranking probabilities
        x_hat_prob = torch.softmax(x_hat_recon, dim=1)

        # Category embedding placeholder
        if matrix_F is not None:
            x_hat_cate = torch.zeros(
                x_t.size(0),               # batch size
                self.num_cate,             # number of categories
                matrix_F.size(1),          # number of items
                device=x_t.device
            )
        else:
            x_hat_cate = torch.zeros(
                x_t.size(0),
                self.num_cate,
                x_t.size(1),
                device=x_t.device
            )

        # Dummy orthogonal loss
        loss_ortho = torch.zeros(1, device=x_t.device)

        return x_hat_recon, x_hat_prob, x_hat_cate, loss_ortho
