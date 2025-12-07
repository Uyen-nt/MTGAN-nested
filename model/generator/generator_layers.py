import torch
from torch import nn
import torch.nn.functional as F
from model.utils import MaskedAttention


class GRU(nn.Module):
    def __init__(self, code_num, hidden_dim, max_len, device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.device = device

        self.gru_cell = nn.GRUCell(input_size=code_num, hidden_size=hidden_dim)
        self.hidden2codes = nn.Sequential(
            nn.Linear(hidden_dim, code_num),
            nn.Sigmoid()
        )

    def step(self, x, h=None):
        h_n = self.gru_cell(x, h)
        codes = self.hidden2codes(h_n)
        return codes, h_n

    def forward(self, noise):
        codes = self.hidden2codes(noise)
        h = torch.zeros(len(codes), self.hidden_dim, device=self.device)
        samples, hiddens = [], []
        for _ in range(self.max_len):
            samples.append(codes)
            codes, h = self.step(codes, h)
            hiddens.append(h)
        samples = torch.stack(samples, dim=1)
        hiddens = torch.stack(hiddens, dim=1)

        return samples, hiddens


class SmoothCondition(nn.Module):
    def __init__(self, code_num, attention_dim):
        super().__init__()
        self.attention = MaskedAttention(code_num, attention_dim)

    def forward(self, x, lens, target_codes):
        score = self.attention(x, lens)
        score_tensor = torch.zeros_like(x)
        score_tensor[torch.arange(len(x)), :, target_codes] = score
        x = x + score_tensor
        x = torch.clip(x, max=1)
        return x



import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================
# 1. MiniSelfAttention on latent D
# ================================
class MiniSelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        attn = torch.einsum("btd, bTd -> btT", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("btT, bTd -> btd", attn, v)
        return self.proj(out)


# ================================
# 2. CMS-lite on latent D (FAST)
# ================================
class CMSLite(nn.Module):
    def __init__(self, dim, hidden=4):
        super().__init__()
        h = dim * hidden
        self.net = nn.Sequential(
            nn.Linear(dim, h),
            nn.GELU(),
            nn.Linear(h, dim)
        )

    def forward(self, x):
        return self.net(x)


# ================================
# 3. Modifier-lite on latent D
# ================================
class SelfModifierLite(nn.Module):
    def __init__(self, dim, hidden=4):
        super().__init__()
        h = dim * hidden
        self.net = nn.Sequential(
            nn.Linear(dim * 2, h),
            nn.GELU(),
            nn.Linear(h, dim)
        )

    def forward(self, context, cond):
        concat = torch.cat([context, cond], dim=-1)
        return self.net(concat)


# ======================================================
# 4. SmoothCondition_HOPE_FAST (with Projection V <-> D)
# ======================================================
class SmoothCondition_HOPE(nn.Module):
    def __init__(self, code_num, latent_dim=256, hidden=4):
        super().__init__()

        self.code_num = code_num
        self.latent_dim = latent_dim

        # Project ICD vector V → latent D
        self.project_down = nn.Linear(code_num, latent_dim)

        # Project back latent D → V
        self.project_up = nn.Linear(latent_dim, code_num)

        # HOPE-lite modules now run on D, not V
        self.attn = MiniSelfAttention(latent_dim)
        self.cms = CMSLite(latent_dim, hidden)
        self.modifier = SelfModifierLite(latent_dim, hidden)

        # Target code embedding (on latent)
        self.target_embed = nn.Embedding(code_num, latent_dim)

    def forward(self, x, lens, target_codes):
        """
        x: (B, T, V)
        target_codes: (B,)
        """

        B, T, V = x.shape

        # ------------------------
        # 1) Project V → D
        # ------------------------
        h = self.project_down(x)   # (B, T, D)

        # ------------------------
        # 2) Attention on latent D
        # ------------------------
        h_attn = self.attn(h)

        # ------------------------
        # 3) CMS on latent D
        # ------------------------
        h_cms = self.cms(h_attn)

        # ------------------------
        # 4) Add learned embedding for target ICD code
        # ------------------------
        t = self.target_embed(target_codes)    # (B, D)
        t = t.unsqueeze(1).expand(B, T, self.latent_dim)

        h_cond = h_cms + t

        # ------------------------
        # 5) Modifier
        # ------------------------
        h_mod = self.modifier(h_attn, h_cond)

        # ------------------------
        # 6) Residual combine
        # ------------------------
        h_out = h + h_cond + h_mod

        # ------------------------
        # 7) Project back to ICD V
        # ------------------------
        out = self.project_up(h_out)
        out = out.sigmoid()  # same output type as original

        return out
