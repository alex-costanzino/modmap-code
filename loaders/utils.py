import ast
import torchvision
import random
import numpy as np
import torch

class DotDict(dict):
    """A dictionary that supports dot notation and nested dot access, and converts numeric and tuple-like strings to native types."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    def update(self, *args, **kwargs):
        """Override update to recursively convert items."""
        for k, v in dict(*args, **kwargs).items():
            self[k] = self._convert(v)

    def _convert(self, value):
        if isinstance(value, dict):
            return DotDict(value)
        elif isinstance(value, list):
            return [self._convert(i) for i in value]
        elif isinstance(value, str):
            # Try to parse tuples
            if value.startswith("(") and value.endswith(")"):
                try:
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, tuple):
                        # Recursively convert tuple items
                        return tuple(self._convert(i) for i in parsed)
                except Exception:
                    pass
            # Try to parse numbers (int or float)
            try:
                if '.' in value or 'e' in value.lower():
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                # Not a number, return as-is
                return value
        else:
            return value

class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return torchvision.transforms.functional.pad(image, padding, padding_mode = 'edge')
    
def denormalize(image):
    image = torch.clamp(image, 0.0, 1.0).permute(1,2,0)
    return image

def set_seed(seed: int = 42):
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False