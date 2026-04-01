import os
import torch

REPO_DIR_2 = 'dinov2'
WEIGHTS_DIR_2 = 'checkpoints/dino-v2/dinov2_vitb14_reg4_pretrain.pth'

class FeatureExtractor_v2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.fe = torch.hub.load(REPO_DIR_2, 'dinov2_vitb14_reg', source='local', pretrained=False)

        state_dict = torch.load(WEIGHTS_DIR_2, map_location='cpu')
        self.fe.load_state_dict(state_dict, strict=True)

        self.patch_size = self.fe.patch_size
        self.embed_dim = self.fe.embed_dim

    def forward(self, x):
        return self.fe.forward_features(x)['x_norm_patchtokens']

from dinov3_depth.dinov3.models.vision_transformer import vit_small

WEIGHTS_PATH = 'checkpoints/dino-depth/depth_encoder'

import yaml
from loaders.utils import DotDict
with open(os.path.join(WEIGHTS_PATH, 'config.yaml')) as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)
    cfg = DotDict(**conf)

vit_kwcfg = dict(
    patch_size=cfg.student.patch_size,
    pos_embed_rope_base=cfg.student.pos_embed_rope_base,
    pos_embed_rope_min_period=cfg.student.pos_embed_rope_min_period,
    pos_embed_rope_max_period=cfg.student.pos_embed_rope_max_period,
    pos_embed_rope_normalize_coords=cfg.student.pos_embed_rope_normalize_coords,
    pos_embed_rope_shift_coords=cfg.student.pos_embed_rope_shift_coords,
    pos_embed_rope_jitter_coords=cfg.student.pos_embed_rope_jitter_coords,
    pos_embed_rope_rescale_coords=cfg.student.pos_embed_rope_rescale_coords,
    qkv_bias=cfg.student.qkv_bias,
    layerscale_init=cfg.student.layerscale,
    norm_layer=cfg.student.norm_layer,
    ffn_layer=cfg.student.ffn_layer,
    ffn_bias=cfg.student.ffn_bias,
    proj_bias=cfg.student.proj_bias,
    n_storage_tokens=cfg.student.n_storage_tokens,
    mask_k_bias=cfg.student.mask_k_bias,
    untie_cls_and_patch_norms=cfg.student.untie_cls_and_patch_norms,
    untie_global_and_local_cls_norm=cfg.student.untie_global_and_local_cls_norm,
)

class FeatureExtractorDepth(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.fe = vit_small(**vit_kwcfg)
        self.weights = torch.load(os.path.join(WEIGHTS_PATH, 'eval/last_checkpoint/teacher_checkpoint.pth'))['teacher']

        self.weights = {k.replace("backbone.", ""): v for k, v in self.weights.items() if k.startswith("backbone.")}

        self.fe.load_state_dict(self.weights)
        print(f'Loaded weights from: {WEIGHTS_PATH}.')

        self.patch_size = self.fe.patch_size
        self.embed_dim = self.fe.embed_dim

    def forward(self, x):
        return self.fe.forward_features(x)['x_norm_patchtokens']
