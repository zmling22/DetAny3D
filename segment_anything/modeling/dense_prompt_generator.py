import torch
import torch.nn as nn

class DensePromptGenerator(nn.Module):
    def __init__(self,
                 num_points=50,
                 embed_dim=256,
                 num_heads=8,
                 use_mlp=True,
                 mlp_hidden_dim=128):
                 
        super().__init__()
        self.num_points = num_points
        self.embed_dim = embed_dim
        
        # 可学习的2D坐标 [N, 2], 每行形如 (x, y)
        self.point_coords = nn.Parameter(torch.randn(num_points, 2))
        
        # 把 scalar x / y 分别编码成 embed 的小网络
        # 这里写成一个简单 MLP: scalar -> hidden -> embed_dim
        self.x_mlp = nn.Sequential(
            nn.Linear(1, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, embed_dim),
        )
        self.y_mlp = nn.Sequential(
            nn.Linear(1, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, embed_dim),
        )
        
        # 简单的自注意力层，这里只写一层
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        
    def forward(self):
        """
        Return:
            prompt_embeddings: shape [N, 2, C]
                其中 2 代表每个点的 (x_token, y_token)
        """
        # self.point_coords: [N, 2]
        coords = self.point_coords  # (N, 2)
        
        # 分别取 x, y 两列 => [N, 1]
        x_val = coords[:, 0:1]  # shape (N, 1)
        y_val = coords[:, 1:2]  # shape (N, 1)
        
        # 分别经过 x_mlp / y_mlp => (N, C)
        x_embed = self.x_mlp(x_val)
        y_embed = self.y_mlp(y_val)
        
        # 堆叠 => [N, 2, C]
        # 这里 dim=1 表示把 x_embed, y_embed 作为同一个维度上的两个 token
        prompt_embeddings = torch.stack([x_embed, y_embed], dim=1)  # (N, 2, C)

        # =============== 做自注意力 ===============
        # 1) 先 reshape 成 [N*2, C] 作为“序列长度 N*2，batch=1”的形式
        #    PyTorch MHA 默认输入形状: [seq_len, batch, embed_dim]
        tokens = prompt_embeddings.reshape(-1, self.embed_dim)         # => (N*2, C)
        tokens = tokens.unsqueeze(1)  # => (N*2, 1, C)   # batch = 1
        
        # 2) 做 Self-Attention
        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        # attn_out: (N*2, 1, C)
        
        # 3) reshape 回 [N, 2, C]
        attn_out = attn_out.squeeze(1).view(self.num_points, 2, self.embed_dim)
        
        return attn_out  # [N, 2, C]