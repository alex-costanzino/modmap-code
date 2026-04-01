import os
import torch
import numpy as np
import cv2
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import itertools

def class_labels():
    return [
        "plastic_stool",
        "rubbish_bin",
        "wicker_vase",
        "bathroom_forniture",
        "container",
        "plastic_vase",
        "sink_cabinet",
        "wooden_stool",
    ]

def one_hot_lookup():
    labels = class_labels()
    n = len(labels)
    eye = torch.eye(n)
    return {label: eye[i] for i, label in enumerate(labels)}

lookup = one_hot_lookup()

class SquarePad:
    def __call__(self, image):
        if image.dtype != torch.float32:
            image = image.float()
        _, h, w = image.shape
        max_wh = max(h, w)
        pad_left = (max_wh - w) // 2
        pad_right = max_wh - w - pad_left
        pad_top = (max_wh - h) // 2
        pad_bottom = max_wh - h - pad_top
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        return transforms.functional.pad(image, padding, padding_mode='edge')


class RemoveInf():
    def __call__(self, image):
        image[image == np.inf] = 0.0
        return image
    
class RemoveMax():
    def __call__(self, image):
        image[image == image.max()] = 0.0
        return image


class BaseAnomalyDetectionDatasetDepth(Dataset):
    def __init__(self, class_name, img_size, dataset_path):
        self.IMAGENET_MEAN = [0.445]
        self.IMAGENET_STD = [0.269]

        self.cls = class_name
        self.size = img_size
        self.dataset_path = dataset_path

        self.cls_path = os.path.join(self.dataset_path, self.cls)
        
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            SquarePad(),
            transforms.Resize((self.size, self.size), interpolation = transforms.InterpolationMode.BICUBIC),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Lambda(lambda img: img / img.max()),
            transforms.Normalize(mean = self.IMAGENET_MEAN, std = self.IMAGENET_STD)
            ])
        
        self.depth_transform = transforms.Compose([
            transforms.ToTensor(),
            RemoveInf(),
            SquarePad(),
            transforms.Resize((self.size, self.size), interpolation = transforms.InterpolationMode.NEAREST),
            transforms.Lambda(lambda img: img / img.max()),
            ])
        
        self.depth_synth_transform = transforms.Compose([
            transforms.ToTensor(),
            RemoveMax(),
            SquarePad(),
            transforms.Resize((self.size, self.size), interpolation = transforms.InterpolationMode.NEAREST),
            transforms.Lambda(lambda img: img / img.max()),
            ])
        
        self.metadata = None

class TrainRealDataset(BaseAnomalyDetectionDatasetDepth):
    def __init__(self, class_name, img_size, dataset_path):
        super().__init__(class_name = class_name, img_size = img_size, dataset_path = dataset_path)

        self.load_dataset()
        self.num_views = len(self.view_ids)

    def load_dataset(self):
        depth_paths = glob.glob(os.path.join(self.cls_path, self.cls + '_real', 'depth') + "/*.npy")
        rgb_paths = [path.replace('.npy', '_2.png').replace('depth', 'rgb') for path in depth_paths]
        
        rgb_paths.sort(), depth_paths.sort()

        self.view_ids = [f'C{idx+1}' for idx, _ in enumerate(rgb_paths)]

        self.data = {
            idx: {
                "rgb_path": rgb_path,
                "depth_path": depth_path,
            }
            for idx, rgb_path, depth_path in zip(self.view_ids, rgb_paths, depth_paths)
        }
        
        self.pairs = [(i, j) for i, j in itertools.product(self.view_ids, repeat=2)] # Cross-pairs and self-pairs.

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        vsrc, vtrg = self.pairs[idx]

        def load(view_id):
            rgb = cv2.imread(self.data[view_id]['rgb_path'], cv2.IMREAD_GRAYSCALE)
            depth = np.load(self.data[view_id]['depth_path'])

            rgb = self.rgb_transform(rgb)
            depth = self.depth_transform(depth)

            return rgb, depth

        rgb_source, depth_source = load(vsrc)
        rgb_target, depth_target = load(vtrg)

        onehot_src = torch.nn.functional.one_hot(torch.tensor(int(vsrc[1:]) - 1), self.num_views).float()
        onehot_trg = torch.nn.functional.one_hot(torch.tensor(int(vtrg[1:]) - 1), self.num_views).float()

        return rgb_source, depth_source, onehot_src, rgb_target, depth_target, onehot_trg, lookup[self.cls]

class TrainSynthDataset(BaseAnomalyDetectionDatasetDepth):
    def __init__(self, class_name, img_size, dataset_path):
        super().__init__(class_name = class_name, img_size = img_size, dataset_path = dataset_path)

        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

        self.load_dataset()
        self.num_views = len(self.view_ids)

    def load_dataset(self):
        depth_paths = glob.glob(os.path.join(self.cls_path, self.cls + '_synth', 'DEPTH') + "/*.exr")
        rgb_paths = [path.replace('.exr', '.png').replace('DEPTH', 'RGB').replace('_depth', '_2') for path in depth_paths]

        rgb_paths.sort(), depth_paths.sort()

        self.view_ids = [f'C{idx+1}' for idx, _ in enumerate(rgb_paths)]

        self.data = {
            idx: {
                "rgb_path": rgb_path,
                "depth_path": depth_path,
            }
            for idx, rgb_path, depth_path in zip(self.view_ids, rgb_paths, depth_paths)
        }
        
        self.pairs = [(i, j) for i, j in itertools.product(self.view_ids, repeat=2)] # Cross-pairs and self-pairs.

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        vsrc, vtrg = self.pairs[idx]

        def load(view_id):
            rgb = cv2.imread(self.data[view_id]['rgb_path'], cv2.IMREAD_GRAYSCALE)
            depth = cv2.imread(self.data[view_id]['depth_path'], cv2.IMREAD_UNCHANGED)

            rgb = self.rgb_transform(rgb)
            depth = self.depth_synth_transform(depth[...,0])
            depth[depth < 0.0] = 0.0

            return rgb, depth

        rgb_source, depth_source = load(vsrc)
        rgb_target, depth_target = load(vtrg)

        onehot_src = torch.nn.functional.one_hot(torch.tensor(int(vsrc[1:]) - 1), self.num_views).float()
        onehot_trg = torch.nn.functional.one_hot(torch.tensor(int(vtrg[1:]) - 1), self.num_views).float()

        return rgb_source, depth_source, onehot_src, rgb_target, depth_target, onehot_trg, lookup[self.cls]

class TestDataset(BaseAnomalyDetectionDatasetDepth):
    def __init__(self, class_name, img_size, dataset_path):
        super().__init__(class_name = class_name, img_size = img_size, dataset_path = dataset_path)

        self.load_dataset()

    def load_dataset(self):
        sub_folders = os.listdir(self.cls_path)
        sub_folders = [f for f in sub_folders if 'real' not in f and 'synth' not in f]
        sub_folders.sort()

        self.data = []

        for sub_folder in sub_folders:
            depth_paths = glob.glob(os.path.join(self.cls_path, sub_folder, 'depth') + "/*.npy")
            rgb_paths = [path.replace('.npy', '_2.png').replace('depth', 'rgb') for path in depth_paths]
            rgb_paths.sort(), depth_paths.sort()

            if 'bad' in sub_folder:
                label = 1.0
            elif 'bad' not in sub_folder:
                label = 0.0

            view_ids = [f'C{idx+1}' for idx, _ in enumerate(rgb_paths)]
            self.num_views = len(view_ids)

            data = {
                idx: {
                    "rgb_path": rgb_path,
                    "depth_path": depth_path,
                    "label" : label
                }
                for idx, rgb_path, depth_path in zip(view_ids, rgb_paths, depth_paths)
            }

            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        for view in self.data[idx].keys():
            rgb = cv2.imread(self.data[idx][view]['rgb_path'], cv2.IMREAD_GRAYSCALE)
            depth = np.load(self.data[idx][view]['depth_path'])

            rgb = self.rgb_transform(rgb)
            depth = self.depth_transform(depth)

            self.data[idx][view]['rgb'] = rgb
            self.data[idx][view]['depth'] = depth
            self.data[idx][view]['one_hot'] = lookup[self.cls]

        return self.data[idx]

def get_data_loader(split, class_name, dataset_path, img_size, batch_size, shuffle = False):
    if split == 'train_real':
        dataset = TrainRealDataset(class_name = class_name, img_size = img_size, dataset_path = dataset_path)
    elif split == 'train_synth':
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        dataset = TrainSynthDataset(class_name = class_name, img_size = img_size, dataset_path = dataset_path)
    elif split == 'test':
        dataset = TestDataset(class_name = class_name, img_size = img_size, dataset_path = dataset_path)

    data_loader = DataLoader(
        dataset = dataset, batch_size = batch_size, shuffle = shuffle, 
        num_workers = 1, drop_last = False, pin_memory = True
        )
    
    return data_loader
