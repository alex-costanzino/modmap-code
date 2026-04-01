import torch
    
class FeatureMapping(torch.nn.Module):
    def __init__(self, in_features = None, out_features = None, act_layer = torch.nn.GELU):
        super().__init__()
        
        self.act_fcn = act_layer()

        self.input = torch.nn.Linear(in_features, (in_features + out_features) // 2)
        self.projection = torch.nn.Linear((in_features + out_features) // 2, (in_features + out_features) // 2)
        self.output = torch.nn.Linear((in_features + out_features) // 2, out_features)

    def forward(self, x):
        x = self.input(x)
        x = self.act_fcn(x)

        x = self.projection(x)
        x = self.act_fcn(x)

        x = self.output(x)

        return x

class FeatureAttentionMapping(torch.nn.Module):
    def __init__(self, input_dim, num_heads=8, num_layers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(
                embed_dim = input_dim, 
                num_heads = num_heads, 
                batch_first = True
                )
            for _ in range(num_layers)
        ])

        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(input_dim) for _ in range(num_layers)])
        
        self.ffns = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, input_dim*4),
                torch.nn.ReLU(),
                torch.nn.Linear(input_dim*4, input_dim)
            ) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        # Expected x shape: [batch_size, seq_len, features]
        for attn, norm, ffn in zip(self.layers, self.norms, self.ffns):
            # Self-attention.
            attn_out, _ = attn(x, x, x)
            x = norm(x + attn_out)
            
            # Feed-forward.
            ff_out = ffn(x)
            x = norm(x + ff_out)
            
        return x