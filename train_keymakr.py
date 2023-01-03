import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
# from synthetic_agumentations_dataloader import SyntheticAugmentationsDataset, collate_fn
# from os2d_dataloader import OS2DDataset, os2d_collate_fn
from keymakr_dataloader import LITWDataset, os2d_collate_fn
import torch
from torch.utils.data import DataLoader
from os2d.config import cfg
from os2d.utils import set_random_seed, get_trainable_parameters, setup_logger
from os2d.modeling.model import build_os2d_from_config
from os2d.engine.optimization import create_optimizer, setup_lr
import math
from collections import OrderedDict
import time
from contextlib import redirect_stdout
import wandb
import os
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


wandb.login()
reference_images_path = "../../data/keymakr-os2d/assets/"
logos_path = "../../data/keymakr-os2d/logos/"
annotations_path = "../../data/keymakr-os2d/train_annotations.csv"

reference_images_val_path = "../../data/keymakr-os2d/assets/"
logos_val_path = "../../data/keymakr-os2d/logos/"
annotations_val_path = "../../data/keymakr-os2d/test_annotations.csv"

i_iter = 0


def checkpoint_model(net, optimizer, log_path, is_cuda, experiment_name=None, model_name=None, i_iter=None,
                     extra_fields=None):
    net.cpu()
    try:
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        checkpoint = {}
        checkpoint["net"] = net.state_dict()
        checkpoint["optimizer"] = optimizer.state_dict()
        if extra_fields:
            checkpoint.update(extra_fields)
        if experiment_name:
            checkpoint_file_name = f"checkpoint_{experiment_name}_{i_iter}.pth"
        elif model_name:
            checkpoint_file_name = f"checkpoint_{model_name}.pth"
        else:
            checkpoint_file_name = f"checkpoint_iter_{i_iter}.pth"
        checkpoint_file = os.path.join(log_path, checkpoint_file_name)
        torch.save(checkpoint, checkpoint_file)
        return checkpoint_file
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        print("Could not save the checkpoint model for some reason: {}".format(str(e)))

    if is_cuda:
        net.cuda()


@torch.no_grad()
def evaluate(eval_dataloader, net, box_coder, optimizer, criterion):
    print("Evaluating...")
    eval_losses = []
    net.eval()
    for i, batch_data in enumerate(eval_dataloader):
        if i % 50 == 0:
            print(i)
        images, class_images, loc_targets, class_targets, class_ids, class_image_sizes, \
        batch_box_inverse_transform, batch_boxes, batch_img_size, _, _ = batch_data

        if cfg.is_cuda:
            net = net.cuda()
            images = images.cuda()
            class_images = [img.cuda() for img in class_images]
            loc_targets = loc_targets.cuda()
            class_targets = class_targets.cuda()

        loc_scores, class_scores, class_scores_transform_detached, fm_sizes, corners = \
            net(images, class_images,
                train_mode=True,
                fine_tune_features=cfg.train.model.train_features)
        cls_targets_remapped, ious_anchor, ious_anchor_corrected = \
            box_coder.remap_anchor_targets(loc_scores, batch_img_size, class_image_sizes, batch_boxes)

        losses = criterion(loc_scores, loc_targets,
                           class_scores, class_targets,
                           cls_targets_remapped=cls_targets_remapped,
                           cls_preds_for_neg=class_scores_transform_detached if not cfg.train.model.train_transform_on_negs else None)

        eval_losses.append(losses["loss"].item())
        # wandb.log({"eval_loss": np.mean(eval_losses)})

    return np.mean(eval_losses)


def train_epoch(train_dataloader, net, box_coder, optimizer, criterion):  # , anneal_lr_func):
    global i_iter
    net.train(freeze_bn_in_extractor=cfg.train.model.freeze_bn,
              freeze_transform_params=cfg.train.model.freeze_transform,
              freeze_bn_transform=cfg.train.model.freeze_bn_transform)
    train_losses = []
    times = []
    for batch_data in train_dataloader:
        time0 = time.time()
        i_iter += 1
        if i_iter % 50 == 0:
            avg = np.mean(times)
            print(f"Iteration {i_iter} {avg}")
        images, class_images, loc_targets, class_targets, class_ids, class_image_sizes, \
        batch_box_inverse_transform, batch_boxes, batch_img_size, _, _ = batch_data

        if cfg.is_cuda:
            net = net.cuda()
            images = images.cuda()
            class_images = [img.cuda() for img in class_images]
            loc_targets = loc_targets.cuda()
            class_targets = class_targets.cuda()

        optimizer.zero_grad(set_to_none=True)
        # print(images.shape)
        loc_scores, class_scores, class_scores_transform_detached, fm_sizes, corners = \
            net(images, class_images,
                train_mode=True,
                fine_tune_features=cfg.train.model.train_features)

        cls_targets_remapped, ious_anchor, ious_anchor_corrected = \
            box_coder.remap_anchor_targets(loc_scores, batch_img_size, class_image_sizes, batch_boxes)

        losses = criterion(loc_scores, loc_targets,
                           class_scores, class_targets,
                           cls_targets_remapped=cls_targets_remapped,
                           cls_preds_for_neg=class_scores_transform_detached if not cfg.train.model.train_transform_on_negs else None)

        main_loss = losses["loss"]
        main_loss.backward()

        # lr = anneal_lr_func(i_iter + 1, anneal_now=i_iter > cfg.train.optim.anneal_lr.initial_patience)

        # if cfg.train.optim.anneal_lr.reload_best_model_after_anneal_lr and lr != get_learning_rate(optimizer):
        #    print("Annealing...")
        #    set_learning_rate(optimizer, lr)

        train_losses.append(main_loss.item() / cfg.train.batch_size)
        # wandb.log({"train_loss": np.mean(train_losses)})
        # save full grad
        grad = OrderedDict()
        for name, param in net.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad[name] = param.grad.clone().cpu()

        grad_norm = torch.nn.utils.clip_grad_norm_(get_trainable_parameters(net), cfg.train.optim.max_grad_norm,
                                                   norm_type=2)

        if math.isnan(grad_norm):
            print("gradient is NaN")
        else:
            optimizer.step()

        # save intermediate model
        # if cfg.output.path and cfg.output.save_iter and i_iter % cfg.output.save_iter == 0:
        #    print("Saving...")
        #    checkpoint_model(net, optimizer, cfg.output.path, cfg.is_cuda, i_iter=i_iter, experiment_name=wandb.run.name)
        time1 = time.time()
        times.append(time1 - time0)
        time0 = time1
    checkpoint_model(net, optimizer, cfg.output.path, cfg.is_cuda, i_iter=i_iter, experiment_name=wandb.run.name)

    return np.mean(train_losses)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    cfg.init.model = "best_os2d_checkpoint.pth"
    # cfg.init.model = "keymakr_cpts/checkpoint_honest-thunder-44_29838.pth"
    # cfg.init.model = "keymakr_cpts/checkpoint_lunar-breeze-45_29838.pth"
    # cfg.init.model = "keymakr_cpts/checkpoint_confused-sponge-46_19892.pth"
    # cfg.init.model = "keymakr_cpts/checkpoint_hearty-snowball-47_29838.pth"
    # cfg.init.model = "keymakr_cpts/checkpoint_earnest-spaceship-90_9946.pth"
    # cfg.init.model = "keymakr_cpts/checkpoint_breezy-voice-95_9946.pth"
    # cfg.init.model = "keymakr_cpts/checkpoint_rare-yogurt-97_9946.pth"
    # cfg.init.model = "keymakr_cpts/checkpoint_honest-plant-98_9946.pth"
    # cfg.init.model = "keymakr_cpts/checkpoint_honest-plant-98_9946_1.pth"
    # cfg.init.model = "keymakr_cpts/checkpoint_honest-plant-98_9946_2.pth"
    # cfg.init.model = "keymakr_cpts/checkpoint_swift-paper-100_9946_1.pth"
    # cfg.init.model = "keymakr_cpts/checkpoint_swift-paper-100_9946_2.pth"
    # cfg.init.model = "keymakr_cpts/checkpoint_swift-paper-100_9946_3.pth"
    # cfg.init.model = "keymakr_cpts/checkpoint_winter-violet-104_29838_2.pth"
    # cfg.init.model = "keymakr_cpts/checkpoint_pious-dust-102_6220_1.pth"

    cfg.is_cuda = torch.cuda.is_available()
    cfg.train.batch_size = 8
    cfg.num_epochs = 1
    cfg.output.path = "keymakr_cpts"
    cfg.output.save_iter = 1000
    cfg.random_seed = 42
    cfg.train.optim.lr = 1e-4
    with open('cfg.yml', 'w') as f:
        with redirect_stdout(f): print(cfg.dump())

    config = {'num_epochs': cfg.num_epochs,
              'batch_size': cfg.train.batch_size,
              'random_seed': cfg.random_seed,
              'learning_rate': cfg.train.optim.lr,
              'using_all_logos': False,
              'using_dominant_color': True,
              'init_model': cfg.init.model
              }

    wandb.init(project="os2d-keymakr10k", tags=['dominant color + batch size 8 + scale loss'],
               config=config, resume="allow")
    # set this to use faster convolutions
    if cfg.is_cuda:
        assert torch.cuda.is_available(), "Do not have available GPU, but cfg.is_cuda == 1"
        torch.backends.cudnn.benchmark = True

    # random seed
    set_random_seed(cfg.random_seed, cfg.is_cuda)

    net, box_coder, criterion, img_normalization, optimizer_state = build_os2d_from_config(cfg)
    parameters = get_trainable_parameters(net)
    optimizer = create_optimizer(parameters, cfg.train.optim, optimizer_state)
    # _, anneal_lr_func = setup_lr(optimizer, full_log, cfg.train.optim.anneal_lr, cfg.eval.iter)

    # train_dataset = SyntheticAugmentationsDataset(reference_images_path, logos_path, box_coder)
    train_dataset = LITWDataset(reference_images_path, logos_path, annotations_path, box_coder)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0,
                                  collate_fn=os2d_collate_fn)

    eval_dataset = LITWDataset(reference_images_val_path, logos_val_path, annotations_val_path, box_coder)
    eval_dataloader = DataLoader(eval_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0,
                                 collate_fn=os2d_collate_fn)
    train_losses = []
    for i in range(cfg.num_epochs):
        torch.cuda.empty_cache()

        train_loss = train_epoch(train_dataloader, net, box_coder, optimizer, criterion)  # , anneal_lr_func)
        wandb.log({'train_loss' : train_loss})

        eval_loss = evaluate(eval_dataloader, net, box_coder, optimizer, criterion)
        wandb.log({'eval_loss' : eval_loss})

        print(train_loss, eval_loss)

    print("done")
