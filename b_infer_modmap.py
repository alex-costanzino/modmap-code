import os
import torch
import numpy
import cv2

import yaml
from loaders.utils import DotDict
with open("configs/config.yaml") as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)
    cfg = DotDict(**conf)

import matplotlib.pyplot as plt

from loaders.sim3d_loader import TestDataset, class_labels
from loaders.utils import denormalize, set_seed

from models.feature_mapping import FeatureMapping
from models.dino_features import FeatureExtractor_v3, FeatureExtractor_v2, FeatureExtractorDepth
from models.film_modulator import Modulator

def filter_mask(mask: torch.Tensor, kernel_size: int = 3, iterations: int = 1) -> torch.Tensor:
    """
    Apply opening operation to the mask then suppress spurious connected components.
    Input:  mask (1,1,H,W) bool Tensor
    Output: mask (1,1,H,W) bool Tensor
    """
    assert mask.ndim == 4 and mask.shape[0] == 1 and mask.shape[1] == 1
    assert mask.dtype == torch.bool

    device = mask.device

    # Force binary uint8 on CPU.
    mask_np = (mask[0, 0].to(torch.uint8).cpu().numpy() > 0).astype(numpy.uint8)

    # Step 1: opening.
    if iterations > 0 and kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel, iterations=iterations)

    # Step 2: connected components filtering.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=4)
    if num_labels <= 1:
        out = torch.from_numpy(mask_np).to(device=device, dtype=torch.bool)
        return out.unsqueeze(0).unsqueeze(0)

    # Step 3: keep largest component.
    areas = stats[:, cv2.CC_STAT_AREA].copy()
    areas[0] = 0  # Ignore background.
    largest_label = int(areas.argmax())

    filtered = (labels == largest_label).astype(numpy.uint8)

    return torch.from_numpy(filtered).to(device=device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)

def infer(model_image2depth, modulator_image, model_depth2image, modulator_depth, feature_extractor_image, feature_extractor_depth, sample, target_view_id, num_views, path, metadata):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor_image.to(device), feature_extractor_depth.to(device)
    model_image2depth.to(device), modulator_image.to(device), model_depth2image.to(device), modulator_depth.to(device)

    cos_sim = torch.nn.CosineSimilarity(dim=-1)

    target_view = torch.nn.functional.one_hot(torch.tensor(int(target_view_id[1:]) - 1), num_views).float().to(device).unsqueeze(0)  # [num_views, num_views]
    
    anomaly_maps = []
    image_anomaly_maps = []
    depth_anomaly_maps = []

    rgb_target = sample[target_view_id]['rgb'].to(device)
    depth_target = sample[target_view_id]['depth'].to(device)

    rgb_features_target, depth_features_target = feature_extractor_image(rgb_target), feature_extractor_depth(depth_target)

    inference_time = []
    start_event = torch.cuda.Event(enable_timing = True)
    end_event = torch.cuda.Event(enable_timing = True)

    for source_view_id in sample.keys():
        source_view = torch.nn.functional.one_hot(torch.tensor(int(source_view_id[1:]) - 1), num_views).float().to(device).unsqueeze(0)  # [num_views, num_views]

        rgb_source = sample[source_view_id]['rgb'].to(device)
        depth_source = sample[source_view_id]['depth'].to(device)

        start_event.record()

        rgb_features_source, depth_features_source = feature_extractor_image(rgb_source), feature_extractor_depth(depth_source)

        # Mapping from image to depth space.
        modulated_rgb_features = modulator_image(rgb_features_source, source_view, target_view)
        depth_features_pred = model_image2depth(modulated_rgb_features)
        loss_image2depth = 1 - cos_sim(depth_features_pred, depth_features_target)

        # Mapping from depth to image space.
        modulated_depth_features = modulator_depth(depth_features_source, source_view, target_view)
        image_features_pred = model_depth2image(modulated_depth_features)
        loss_depth2image = 1 - cos_sim(image_features_pred, rgb_features_target)

        end_event.record()
        torch.cuda.synchronize()
        inf_time = start_event.elapsed_time(end_event)

        inference_time.append(inf_time)
        
        anomaly_map = torch.minimum(loss_image2depth, loss_depth2image).cpu().detach()

        anomaly_maps.append(anomaly_map)
        depth_anomaly_maps.append(loss_image2depth.cpu().detach())
        image_anomaly_maps.append(loss_depth2image.cpu().detach())

        if cfg.generate_qualitatives:
            
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

            anomaly_map = anomaly_map.reshape(1, 1, rgb_target.shape[2] // feature_extractor_image.patch_size, rgb_target.shape[3] // feature_extractor_image.patch_size).detach().cpu()
            anomaly_map = torch.nn.functional.interpolate(anomaly_map, size = [rgb_target.shape[2], rgb_target.shape[3]], mode = 'bilinear')
            bg_mask = (depth_target[:,0:1,:,:] != 0.0)
            anomaly_map[~filter_mask(bg_mask)] = 0.0

            plt.imsave(save_path.replace('_am.png', f'_am_{source_view_id}_{target_view_id}.png'), anomaly_map.squeeze(), cmap = 'jet')

        del rgb_source, depth_source, rgb_features_source, depth_features_source, depth_features_pred, image_features_pred, loss_image2depth, loss_depth2image
        torch.cuda.memory.empty_cache()

    print(f"Average inference time per-sample: {numpy.mean(inference_time)*num_views}.")

    depth2image_map = torch.stack(image_anomaly_maps).min(dim=0)[0]
    depth2image_map = depth2image_map.reshape(1, 1, rgb_target.shape[2] // feature_extractor_image.patch_size, rgb_target.shape[3] // feature_extractor_image.patch_size).detach().cpu()
    depth2image_map = torch.nn.functional.interpolate(depth2image_map, size = [rgb_target.shape[2], rgb_target.shape[3]], mode = 'bilinear')
    
    # Filtering out background starting from depth.
    bg_mask = (depth_target[:,0:1,:,:] != 0.0)
    depth2image_map[~filter_mask(bg_mask)] = 0.0
    
    depth2image_map = depth2image_map.squeeze().numpy()

    if cfg.generate_qualitatives:
        
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        numpy.save(save_path.replace('_am.png', '_depth2image.npy'), depth2image_map)
        plt.imsave(save_path.replace('_am.png', '_depth2image.png'), depth2image_map, cmap = 'jet')

    image2depth_map = torch.stack(depth_anomaly_maps).min(dim=0)[0]
    image2depth_map = image2depth_map.reshape(1, 1, rgb_target.shape[2] // feature_extractor_image.patch_size, rgb_target.shape[3] // feature_extractor_image.patch_size).detach().cpu()
    image2depth_map = torch.nn.functional.interpolate(image2depth_map, size = [rgb_target.shape[2], rgb_target.shape[3]], mode = 'bilinear')
    
    # Filtering out background starting from depth.
    bg_mask = (depth_target[:,0:1,:,:] != 0.0)
    image2depth_map[~filter_mask(bg_mask)] = 0.0
    
    image2depth_map = image2depth_map.squeeze().numpy()

    if cfg.generate_qualitatives:
        
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        numpy.save(save_path.replace('_am.png', '_image2depth.npy'), image2depth_map)
        plt.imsave(save_path.replace('_am.png', '_image2depth.png'), image2depth_map, cmap = 'jet')

    final_map = image2depth_map * image2depth_map

    if cfg.generate_qualitatives:
        
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        plt.imsave(save_path, final_map, cmap = 'jet')

if __name__ == "__main__":

    set_seed()

    for label in class_labels():

        cfg.class_name = label

        dataset = TestDataset(
            class_name=cfg.class_name,
            img_size=cfg.data_params.image_size,
            dataset_path=cfg.data_params.dataset_path)

        data_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 1, shuffle = True, num_workers = 8, drop_last = False, pin_memory = False)

        feature_extractor_image = FeatureExtractor_v2()
        feature_extractor_image.eval()

        feature_extractor_depth = FeatureExtractorDepth()
        feature_extractor_depth.eval()

        if cfg.train_params.mapper == 'mlp':
            model_image2depth = FeatureMapping(
                in_features=feature_extractor_image.embed_dim,
                out_features=feature_extractor_depth.embed_dim)
            model_depth2image = FeatureMapping(
                in_features=feature_extractor_depth.embed_dim,
                out_features=feature_extractor_image.embed_dim)

        model_image2depth.eval()
        model_depth2image.eval()

        modulator_image = Modulator(
            n_views=dataset.num_views, 
            hidden_dim=cfg.net_params.modulator_hidden_dim, 
            feat_dim=feature_extractor_image.embed_dim)
        modulator_depth = Modulator(
            n_views=dataset.num_views, 
            hidden_dim=cfg.net_params.modulator_hidden_dim, 
            feat_dim=feature_extractor_depth.embed_dim)

        modulator_image.eval()
        modulator_depth.eval()

        ckpt = torch.load(f"{cfg.data_params.checkpoints_savepath}/modmap_{cfg.exp_name}_{cfg.model}_{cfg.data_params.setup}_{cfg.class_name}.pth", weights_only=False)
        model_image2depth.load_state_dict(ckpt["model_image2depth"])
        model_depth2image.load_state_dict(ckpt["model_depth2image"])
        modulator_image.load_state_dict(ckpt["modulator_image"])
        modulator_depth.load_state_dict(ckpt["modulator_depth"])

        for test_sample in data_loader:
            for view in test_sample.keys():

                save_path = test_sample[view]['rgb_path'][0].replace(cfg.data_params.dataset_path, cfg.data_params.qualitatives_savepath).replace('_2.png', '_am.png')

                infer(
                    model_image2depth=model_image2depth, modulator_image=modulator_image,
                    model_depth2image=model_depth2image, modulator_depth=modulator_depth,
                    feature_extractor_image=feature_extractor_image, feature_extractor_depth=feature_extractor_depth, 
                    sample=test_sample,
                    target_view_id=view,
                    num_views=dataset.num_views,
                    path=save_path,
                    metadata=dataset.metadata
                    )
