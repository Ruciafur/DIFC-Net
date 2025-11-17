#v3
# classifier.py
import torch
import torch.nn as nn

class Classifier(nn.Module):
    """
    输入: x [B, input_dim]   （默认 256）
    内部使用 2-token 的注意力池化（x + learnable context），
    然后再过 MLP -> Logit。
    不改动外部接口，直接替换即可。
    """
    def __init__(self, input_dim: int = 256, num_heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.context = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)
        self.mha = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D]
        """
        B, D = x.shape
        tokens = torch.cat([x.unsqueeze(1), self.context.expand(B, -1, -1)], dim=1)  # [B, 2, D]
        attn_out, _ = self.mha(tokens, tokens, tokens)                                # [B, 2, D]
        pooled = attn_out.mean(dim=1)                                                 # [B, D]
        pooled = self.norm(pooled)
        logit = self.mlp(pooled)                                                      # [B, 1]
        return logit


# #v2
# # classifier.py
# import torch.nn as nn

# class Classifier(nn.Module):
#     """
#     更强的分类头:
#       - LayerNorm 稳定特征分布
#       - GELU + Dropout(0.5)
#       - 两层 MLP
#       - 输出 logits (未做 sigmoid)
#     """
#     def __init__(self, input_dim: int, hidden1: int = 256, hidden2: int = 128, dropout: float = 0.5):
#         super().__init__()
#         self.classifier = nn.Sequential(
#             nn.LayerNorm(input_dim),
#             nn.Linear(input_dim, hidden1),
#             nn.GELU(),
#             nn.Dropout(dropout),

#             nn.Linear(hidden1, hidden2),
#             nn.GELU(),
#             nn.Dropout(dropout),

#             nn.Linear(hidden2, 1)
#         )

#     def forward(self, x):
#         return self.classifier(x)





# import torch.nn as nn

# class Classifier(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 1)
#         )

#     def forward(self, x):
#         return self.classifier(x)