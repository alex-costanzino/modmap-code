import torch
import wandb

import yaml
from loaders.utils import DotDict
with open("configs/config.yaml") as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)
    cfg = DotDict(**conf)

from itertools import chain

from loaders.sim3d_loader import TrainRealDataset, TrainSynthDataset, class_labels
from loaders.utils import set_seed

from models.feature_mapping import FeatureMapping
from models.dino_features import FeatureExtractor_v2, FeatureExtractorDepth
from models.film_modulator import Modulator


def optimise_model(model_image2depth, modulator_image, model_depth2image, modulator_depth, feature_extractor_image, feature_extractor_depth, dataset, lr, epochs, batch_size, save_path):

    wandb.init(
        project='ModMap',
        name='_'.join([cfg.exp_name, cfg.model, cfg.data_params.setup, cfg.class_name]),
        config=conf,
        mode = "disabled"
    )

    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor_image.to(device)
    feature_extractor_depth.to(device)
    model_image2depth.to(device), modulator_image.to(device), model_depth2image.to(device), modulator_depth.to(device)

    cos_sim = torch.nn.CosineSimilarity(dim=-1)

    optimiser_model = torch.optim.Adam([{'params': chain(
        model_image2depth.parameters(), modulator_image.parameters(), 
        model_depth2image.parameters(), modulator_depth.parameters()), 
        'lr': lr}])

    if cfg.train_params.use_scheduler:
        if cfg.train_params.scheduler == 'OneCycleLR':
            scheduler_model = torch.optim.lr_scheduler.OneCycleLR(
                optimiser_model,
                max_lr=lr*5, # Rule of thumb.
                total_steps=epochs * len(data_loader),
                pct_start=cfg.train_params.pct_start,
                anneal_strategy=cfg.train_params.anneal_strategy,
                cycle_momentum=cfg.train_params.cycle_momentum
                )
            
        elif cfg.train_params.scheduler == 'CosineAnnealingLR':
            scheduler_model = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser_model,
                eta_min=lr*1e-2, # Rule of thumb.
                T_max=epochs * len(data_loader),
                )

    print(f'Device: {device}.')

    for epoch in range(epochs):
        total_loss = 0

        for rgbs_source, depths_source, onehot_source, rgbs_target, depths_target, onehot_target in data_loader:
            rgbs_source, depths_source, onehot_source, rgbs_target, depths_target, onehot_target = rgbs_source.to(device), depths_source.to(device), onehot_source.to(device), rgbs_target.to(device), depths_target.to(device), onehot_target.to(device)

            # Mapping from image to depth space.
            with torch.no_grad():
                image_features_source, depth_features_target = feature_extractor_image(rgbs_source), feature_extractor_depth(depths_target)

            modulated_image_features_source = modulator_image(image_features_source, onehot_source, onehot_target)

            depth_features_pred = model_image2depth(modulated_image_features_source)

            loss_image2depth = 1 - cos_sim(depth_features_pred, depth_features_target).mean()

            # Mapping from depth to image space.
            with torch.no_grad():
                depth_features_source, image_features_target = feature_extractor_depth(depths_source), feature_extractor_image(rgbs_target)
            
            modulated_depth_features_source = modulator_depth(depth_features_source, onehot_source, onehot_target)

            image_features_pred = model_depth2image(modulated_depth_features_source)

            loss_depth2image = 1 - cos_sim(image_features_pred, image_features_target).mean()

            loss = loss_image2depth + loss_depth2image
            optimiser_model.zero_grad()
            loss.backward()
            optimiser_model.step()

            if cfg.train_params.use_scheduler:
                scheduler_model.step()

            total_loss += loss.item()

        wandb.log({
            "train/total_loss" : total_loss / len(data_loader),
            "train/lr" : scheduler_model.get_last_lr()[0] if cfg.train_params.use_scheduler else lr,
            })
        
        if epoch % 5 == 0:
            print(f"[{epoch}] Loss: {total_loss / len(data_loader):.5f}")

    # Save model and latents.
    torch.save({
        "model_image2depth": model_image2depth.state_dict(), "model_depth2image": model_depth2image.state_dict(),
        "modulator_image" : modulator_image.state_dict(), "modulator_depth" : modulator_depth.state_dict()
    }, save_path)

    wandb.finish()

if __name__ == "__main__":

    set_seed()

    for label in class_labels():

        cfg.class_name = label

        if cfg.data_params.setup == 'real2real':
            dataset = TrainRealDataset(
                class_name=cfg.class_name,
                img_size=cfg.data_params.image_size,
                dataset_path=cfg.data_params.dataset_path)
        elif cfg.data_params.setup == 'synth2real':
            dataset = TrainSynthDataset(
                class_name=cfg.class_name,
                img_size=cfg.data_params.image_size,
                dataset_path=cfg.data_params.dataset_path)

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

        model_image2depth.train()
        model_depth2image.train()

        modulator_image = Modulator(
            n_views=dataset.num_views, 
            hidden_dim=cfg.net_params.modulator_hidden_dim, 
            feat_dim=feature_extractor_image.embed_dim)
        modulator_depth = Modulator(
            n_views=dataset.num_views, 
            hidden_dim=cfg.net_params.modulator_hidden_dim, 
            feat_dim=feature_extractor_depth.embed_dim)

        modulator_image.train()
        modulator_depth.train()

        optimise_model(
            model_image2depth=model_image2depth, modulator_image=modulator_image,
            model_depth2image=model_depth2image, modulator_depth=modulator_depth,
            feature_extractor_image=feature_extractor_image, feature_extractor_depth=feature_extractor_depth, 
            dataset=dataset,
            lr=cfg.train_params.lr,
            epochs=cfg.train_params.epochs,
            batch_size=cfg.train_params.batch_size,
            save_path=f"{cfg.data_params.checkpoints_savepath}/modmap_{cfg.exp_name}_{cfg.model}_{cfg.data_params.setup}_{cfg.class_name}.pth")
