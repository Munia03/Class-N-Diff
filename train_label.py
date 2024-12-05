# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model, get_checkpoint
from datasets import load_dataset
# from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig
from transformers.utils import ContextManagers
import wandb
import random
import math
from diffusion.classification_model import *
import torch.optim as optim


    

#################################################################################
#                             Training Helper Functions                         #
#################################################################################


DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        if args.resume_from_checkpoint is not None:
            print("Resumeeeeeeee fromm checkpointttt 2##########dekwkndwkljcbjhdbcjkhcbmmmmmmmmmmmmmmmmmmxmxmxmxmxmxmxmmxmxmx")
            experiment_dir = args.resume_from_checkpoint.split("/checkpoints")[0]
            checkpoint_dir = f"{experiment_dir}/checkpoints"
            print("checkpoint dir ", checkpoint_dir)
        else:
            os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
            checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
            os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
        
    # tokenizer = CLIPTokenizer.from_pretrained(
    #     "openai/clip-vit-base-patch32", cache_dir=args.tokenizer_path
    # )
    # text_encoder = CLIPTextModel.from_pretrained(
    #     "openai/clip-vit-base-patch32", cache_dir=args.text_encoder_path
    # )

    # text_encoder_dim = text_encoder.config.hidden_size

    # text_encoder.requires_grad_(False)

    model_encoder = ResNet101()
    in_ch = 2048
    model_classifier1 = ClassificationHead(out_dim=1, in_ch=in_ch)  # Creating main classification head
    # model_classifier2 = ClassificationHead(out_dim=3, in_ch=in_ch)  # Creating main classification head

    # if DP:  # Parallelising if number of GPUs allows
    # model_encoder = nn.DataParallel(model_encoder)
    # model_classifier1 = nn.DataParallel(model_classifier1)
    # model_classifier2 = nn.DataParallel(model_classifier2)

    # model_encoder = model_encoder.to(device)  # Sending feature extractor to GPU
    # model_classifier1 = model_classifier1.to(device)  # Sending classifier head to GPU
    # model_classifier2 = model_classifier2.to(device)

    model_encoder = DDP(model_encoder.to(device), device_ids=[rank])
    model_classifier1 = DDP(model_classifier1.to(device), device_ids=[rank])
    # model_classifier2 = DDP(model_classifier2.to(device), device_ids=[rank])

    criterion1 = nn.BCEWithLogitsLoss()
    # criterion2 = nn.CrossEntropyLoss()

        # Defining main optimizer used accross all models
    cl_optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, (list(model_encoder.parameters()) + 
                                           list(model_classifier1.parameters()))),
        
        lr=0.0003, momentum=0.9)


    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
        # text_encoder_dim = text_encoder_dim
    ).to(device)
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)


    if args.resume_from_checkpoint:
        checkpoint = torch.load(args.resume_from_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        ema.load_state_dict(checkpoint['ema'], strict=True)
        opt.load_state_dict(checkpoint['opt'])
        del checkpoint
        logger.info(f"Using checkpoint: {args.resume_from_checkpoint}")
        print("model loaded succesfully")

        

    

    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="", latent_size=latent_size)  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    # dataset = ImageFolder(args.data_path, transform=transform)


    # sampler = DistributedSampler(
    #     dataset,
    #     num_replicas=dist.get_world_size(),
    #     rank=rank,
    #     shuffle=True,
    #     seed=args.global_seed
    # )

    # loader = DataLoader(
    #     dataset,
    #     batch_size=int(args.global_batch_size // dist.get_world_size()),
    #     shuffle=False,
    #     sampler=sampler,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=True
    # )




    #  load dataset from a folder containing images and metadata.csv
    data_files = {}
    if args.data_path is not None:
        data_files["train"] = os.path.join(args.data_path, "**")
    else:
        raise ValueError(
                f"--data-path' value '{args.data_path}' is missing"
            )
    dataset = load_dataset(
        "imagefolder",
        data_files=data_files,
        cache_dir=args.cache_dir,
    )

    logger.info(f"Dataset contains {len(dataset['train']):,} images ({args.data_path})")

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    print("column namse: ", column_names)
    # print("s", dataset['train'][0])

    # 6. Get the column names for input/target.

    image_column = args.image_column
    if image_column not in column_names:
        raise ValueError(
            f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
        )

    caption_column = args.caption_column
    if caption_column not in column_names:
        raise ValueError(
            f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
        )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                print("wowoowowowoowowwowoow")
                print(type(caption))

                raise ValueError(

                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        return inputs.input_ids


    # Preprocessing the datasets.

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.image_size) if args.center_crop else transforms.RandomCrop(args.image_size),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    genders = {'male': 0, 'female': 1, 'unknown': 1}
    skin_tone = {'lighter': 0, 'brown': 1, 'darker': 2}

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        # examples["input_ids"] = tokenize_captions(examples)
        examples['sex'] = [genders[f] for f in examples['sex']]
        examples['skin_tone'] = [skin_tone[f] for f in examples['skin_tone']]
        examples['label'] = [t for t in examples['target']]
        # examples['age'] = [a for a in examples['age']]
        return examples

    # with accelerator.main_process_first():
    #     if args.max_train_samples is not None:
    #         dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    #     # Set the training transforms

    train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        # input_ids = torch.stack([example["input_ids"] for example in examples])
        sex = torch.tensor([example["sex"] for example in examples])
        skin_tone = torch.tensor([example["skin_tone"] for example in examples])
        label = torch.tensor([example["label"] for example in examples])
        # age = torch.tensor([example["age"] for example in examples])
        return {"pixel_values": pixel_values, "sex": sex, "skin_tone": skin_tone, "label":label}

    # DataLoaders creation:
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     shuffle=True,
    #     collate_fn=collate_fn,
    #     batch_size=args.train_batch_size,
    #     num_workers=args.dataloader_num_workers,
    # )



    sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )

    loader = DataLoader(
        train_dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        collate_fn=collate_fn,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )


    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_loss_cl = 0
    start_time = time()

    first_epoch = 0

    print(len(loader))
    num_update_steps_per_epoch = len(loader)
    print("num_update_steps_per_epoch", num_update_steps_per_epoch)
    # Prepare models for training:
    if args.resume_from_checkpoint:

        train_steps = int((args.resume_from_checkpoint.split("/")[-1]).split(".")[0])

        print("train step ", train_steps)

        first_epoch = math.ceil(train_steps / num_update_steps_per_epoch)

        print("first epoch: ", first_epoch)
    else:
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights

    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(first_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        # for x, y in loader:
        for step, batch in enumerate(loader):
            x0 = batch["pixel_values"]
            # y = batch["input_ids"]
            x0 = x0.to(device)
            # y = y.to(device)
            # print(x0.shape)
            # print(m)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x0).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

            # y = text_encoder(batch["input_ids"])[0] #encoder_hidden_states
            y = batch["label"].to(device)
            y1 = batch["skin_tone"].to(device)
            # y2 = batch["age"]
            y2 = batch["sex"]
            # print(x.shape, y.shape, y)






            model_kwargs = dict(y=y, y1=y1, y2=y2)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            x_pred = diffusion.p_sample(model, x, t, model_kwargs=model_kwargs)
            x_new = x_pred['sample']

            # print("pred", x_new.shape)

            random_bool = torch.rand(1).item() > 0.75

            if random_bool:
                feat_out = model_encoder(x_new)
            else:
                feat_out = model_encoder(x)
            logits1 = model_classifier1(feat_out)  # using the main classifier to get output logits

            # logits2 = model_classifier2(feat_out)

            y = y.unsqueeze(1).type_as(logits1)  # unsqueezing to[batch_size,1] and same dtype as logits
            # target2 = target2.unsqueeze(1).type_as(logits2)  # unsqueezing to[batch_size,1] and same dtype as logits



            # print(target.shape, logits.shape)
            # print(a)
            cl_optimizer.zero_grad()
            loss1 = criterion1(logits1, y)  # calculating loss using categorical crossentorpy
            # loss2 = criterion2(logits2, y1)  # calculating loss using categorical crossentorpy

            # loss = 0.6 * loss1 + 0.4 * loss2
            loss_cl =  loss1 
            if not random_bool:
                
                loss_cl.backward()  # backpropegating to calculate gradients

                cl_optimizer.step() 


            
            loss = loss_dict["loss"].mean() 
            if random_bool:
                    loss = loss + 0.2 * loss_cl
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            running_loss_cl += loss_cl.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_cl_loss = torch.tensor(running_loss_cl / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Classification Loss: {avg_cl_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                running_loss_cl = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)


                    torch.save(
                    {'model_state_dict': model_encoder.state_dict(), 'optimizer_state_dict': cl_optimizer.state_dict(),
                     'epoch': epoch}, os.path.join(checkpoint_dir,f'encoder_all_data_Test{train_steps:07d}.pth'))
                    torch.save(model_classifier1.state_dict(),
                           os.path.join(checkpoint_dir, f'classifier1{train_steps:07d}.pth'))
                    # torch.save(model_classifier2.state_dict(),
                    #        os.path.join(checkpoint_dir, f'classifier2.pth'))
                
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-L/4")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)

    parser.add_argument("--ckpt", type=bool, default=False)
    parser.add_argument("--tokenizer_path", type=str, default="results/tokenizer")
    parser.add_argument("--text_encoder_path", type=str, default="results/text_encoder")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--caption_column", type=str, default="text")
    parser.add_argument(
        "--center_crop", default=True, action="store_true" )
    parser.add_argument(
        "--random_flip", default=True, action="store_true" )



    

    args = parser.parse_args()
    main(args)

