import torch

class Modulator(torch.nn.Module):
    def __init__(self, n_views, hidden_dim=256, feat_dim=768):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2*n_views, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 2*feat_dim)
        )
        self.feat_dim = feat_dim

        # --- custom init for last layer ---
        last = self.mlp[-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()
            last.bias[:feat_dim].fill_(1.0)   # γ-bias = 1
            last.bias[feat_dim:].zero_()      # β-bias = 0

    def forward(self, x_src, v_src, v_trg):
        # x_src: [B, N, feat_dim]
        # v_src, v_trg: [B, n_views] one-hot
        v = torch.cat([v_src, v_trg], dim=-1)             # [B, 2*n_views]
        params = self.mlp(v)                              # [B, 2*feat_dim]
        gamma, beta = params.chunk(2, dim=-1)             # [B, feat_dim]

        gamma = gamma.unsqueeze(1)                        # [B,1,feat_dim]
        beta = beta.unsqueeze(1)
        return gamma * x_src + beta                       # [B,N,feat_dim]

class CameraModulator(torch.nn.Module):
    def __init__(self, hidden_dim, feat_dim):
        super().__init__()

        self.proj = torch.nn.Linear(feat_dim, hidden_dim)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 2*feat_dim)
        )

        # --- custom init for last layer ---
        last = self.mlp[-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()
            last.bias[:feat_dim].fill_(1.0)   # γ-bias = 1
            last.bias[feat_dim:].zero_()      # β-bias = 0

    def forward(self, source_features, source_token, target_token):
        source_token = self.proj(source_token)    # [B, proj_dim]
        target_token = self.proj(target_token)    # [B, proj_dim]
        delta = target_token - source_token
        # source_features: [B, N, feat_dim]
        # source_token, target_token: [B, hidden_dim] projected camera tokens from VGGT
        params = self.mlp(delta)                  # [B, 2*feat_dim]
        gamma, beta = params.chunk(2, dim=-1)     # [B, feat_dim]
        gamma = gamma.unsqueeze(1)                # [B,1,feat_dim]
        beta = beta.unsqueeze(1)
        return gamma * source_features + beta     # [B,N,feat_dim]

class ClassModulator(torch.nn.Module):
    def __init__(self, n_views, n_classes, hidden_dim=256, feat_dim=768):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2*n_views + n_classes, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 2*feat_dim)
        )
        self.feat_dim = feat_dim

        # --- custom init for last layer ---
        last = self.mlp[-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()
            last.bias[:feat_dim].fill_(1.0)   # γ-bias = 1
            last.bias[feat_dim:].zero_()      # β-bias = 0

    def forward(self, x_src, v_src, v_trg, c):
        # x_src: [B, N, feat_dim]
        # v_src, v_trg: [B, n_views] one-hot
        v = torch.cat([v_src, v_trg, c], dim=-1)             # [B, 2*n_views + n_classes]
        params = self.mlp(v)                              # [B, 2*feat_dim]
        gamma, beta = params.chunk(2, dim=-1)             # [B, feat_dim]

        gamma = gamma.unsqueeze(1)                        # [B,1,feat_dim]
        beta = beta.unsqueeze(1)
        return gamma * x_src + beta                       # [B,N,feat_dim]
