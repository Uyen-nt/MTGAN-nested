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



# ================================================================
# 1. Mini Self-Attention (nhẹ để lấy context)
# ================================================================
class MiniSelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.dim = dim

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, T, V)
        B, T, V = x.shape
        H = self.heads
        D = V // H

        q = self.to_q(x).view(B, T, H, D)
        k = self.to_k(x).view(B, T, H, D)
        v = self.to_v(x).view(B, T, H, D)

        scores = torch.einsum("bthd, bThd -> bhtT", q, k) / (D ** 0.5)
        attn = scores.softmax(dim=-1)

        out = torch.einsum("bhtT, bThd -> bthd", attn, v)
        out = out.reshape(B, T, V)
        return self.proj(out)


# ================================================================
# 2. CMS-lite: học "conditional matrix" theo dạng MLP
# ================================================================
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


# ================================================================
# 3. SelfModifier-lite
# ================================================================
class SelfModifierLite(nn.Module):
    def __init__(self, dim, hidden=4):
        super().__init__()
        h = dim * hidden
        self.net = nn.Sequential(
            nn.Linear(dim * 2, h),
            nn.GELU(),
            nn.Linear(h, dim)
        )

    def forward(self, context, error_signal):
        concat = torch.cat([context, error_signal], dim=-1)
        return self.net(concat)


# ================================================================
# 4. SmoothCondition_HOPE — phiên bản HOPE-lite
# ================================================================
class SmoothCondition_HOPE(nn.Module):
    def __init__(self, code_num, attention_dim=4, hidden=4):
        super().__init__()
        self.code_num = code_num

        self.attn = MiniSelfAttention(code_num, heads=4)
        self.cms = CMSLite(code_num, hidden=hidden)
        self.modifier = SelfModifierLite(code_num, hidden=hidden)

    def forward(self, x, lens, target_codes):
        """
        x: (B, T, V)
        lens: (B,)
        target_codes: (B,)
        """

        B, T, V = x.shape

        # ----------------------------------------------------------
        # 1. Self-attention để lấy context semantic
        # ----------------------------------------------------------
        attn_out = self.attn(x)  # (B, T, V)

        # ----------------------------------------------------------
        # 2. CMS-lite dựa trên target_codes:
        #    tạo embedding chỉ số → điều kiện ICD học được
        # ----------------------------------------------------------
        target_embed = F.one_hot(target_codes, num_classes=V).float()  # (B, V)
        target_embed = target_embed.unsqueeze(1).expand(B, T, V)        # (B, T, V)

        cms_out = self.cms(attn_out) + target_embed    # (B, T, V)

        # ----------------------------------------------------------
        # 3. SelfModifier-lite: học hiệu chỉnh boost theo context
        # ----------------------------------------------------------
        modifier = self.modifier(attn_out, cms_out)    # (B, T, V)

        # ----------------------------------------------------------
        # 4. Kết hợp lại
        # ----------------------------------------------------------
        out = x + cms_out + modifier
        out = torch.clamp(out, 0.0, 1.0)

        return out
