import os

from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path

import random
import numpy as np

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.utils import make_grid, save_image

import timm, clip
from torchvision.models import vit_b_16, ViT_B_16_Weights
from pytorch_metric_learning import miners as pml_miners
from pytorch_metric_learning import losses as pml_losses

from models.modules import Prompter
from models.modules import ProbabilisticPromptPool as PromptPool
from domainbed import get_dataset, get_domains_and_classes


parser = ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=None)

parser.add_argument('--model', default='vit_base_patch16_clip_224.openai')
parser.add_argument('--backbone', default='resnet18')
parser.add_argument('--prompt-size', type=int, default=30)

parser.add_argument('--use-rand', action='store_true', default=False)
parser.add_argument('--use-meta', action='store_true', default=False)
parser.add_argument('--use-zero', action='store_true', default=False)
parser.add_argument('--use-unif', action='store_true', default=False)

parser.add_argument('--mix-uniform', action='store_true', default=False)
parser.add_argument('--mix-random', action='store_true', default=False)

parser.add_argument('--num-experts', type=int, default=None)
parser.add_argument('--tune-more', action='store_true', default=False)
parser.add_argument('--ml-loss', type=float, default=0.9)
parser.add_argument('--aug-pt', type=int, default=None)

parser.add_argument('--use-kld', action='store_true', default=False)
parser.add_argument('--norm-experts', action='store_true', default=False)
parser.add_argument('--use-softmax', action='store_true', default=False)
parser.add_argument('--use-tanh', action='store_true', default=False)

parser.add_argument('--dataset', default='pacs')
parser.add_argument('--target', default='p')
parser.add_argument('--batch-size', '-bs', type=int, default=128)
parser.add_argument('--augment', '-A', action='store_true', default=False)

parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--lr', type=float, default=1.0E-4)

parser.add_argument('--early-stop', type=float, default=None)
parser.add_argument('--overwrite', action='store_true', default=False)
parser.add_argument('--log-dir', default='log')
parser.add_argument('--tag', default='__test')
parser.add_argument('--profile', action='store_true', default=False)

CFG = parser.parse_args()
DEVICE = CFG.gpu
SEED = (torch.default_generator.initial_seed() if CFG.seed is None else CFG.seed) % 2**32

if CFG.profile:
    from torch.profiler import profile, ProfilerActivity

torch.cuda.set_device(DEVICE)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# LOGGING
LOG_DIR = Path(CFG.log_dir)
EXP_DIR = LOG_DIR.joinpath(f'{CFG.model}_{CFG.dataset}', CFG.target, CFG.tag)
if EXP_DIR.exists():
    answer = 'y' if CFG.overwrite else None
    while answer not in {'y', 'n'}:
        answer = input('Overwrite? [Y/n] ').strip()
        if len(answer) == 0:
            answer = 'y'
    
    if answer[0].lower() == 'y':
        os.system(f'rm -rf "{EXP_DIR}"')
    else:
        exit(0)
SAMPLE_DIR = EXP_DIR.joinpath('samples')
SAMPLE_DIR.mkdir(parents=True)
        
CKPT_FILENAME = EXP_DIR.joinpath('ckpt.pt')
INIT_SAMPLE_FILENAME = SAMPLE_DIR.joinpath('init.png')
LAST_SAMPLE_FILENAME = SAMPLE_DIR.joinpath('last.png')
BEST_SAMPLE_FILENAME = SAMPLE_DIR.joinpath('best.png')

os.system(f'cp "{__file__}" "{EXP_DIR}"')
writer = SummaryWriter(log_dir=EXP_DIR)

print(CFG)
print(f'{EXP_DIR=}')

# DATA PREPARATION
domains, classes = get_domains_and_classes(CFG.dataset)
src_dataset, tar_dataset, src_domains, tar_domains = \
    get_dataset(CFG.dataset, target=CFG.target, augmentation=CFG.augment, return_domain_names=True)

def seed_worker(worker_id):
    np.random.seed(SEED)
    random.seed(SEED)

g = torch.Generator()
g.manual_seed(SEED)

loader_kwargs = dict(
    batch_size=CFG.batch_size,
    num_workers=10,
    worker_init_fn=seed_worker,
    generator=g,
)
train_loader = DataLoader(src_dataset, shuffle=True, **loader_kwargs)
valid_loader = DataLoader(tar_dataset, shuffle=True, **loader_kwargs)

mean = torch.tensor(tar_dataset.transform.transforms[-1].mean).reshape(-1, 1, 1)
std = torch.tensor(tar_dataset.transform.transforms[-1].std).reshape(-1, 1, 1)

# MODEL & OPTIMIZER
expert_dir = Path('./log').joinpath('experts')
if CFG.use_meta:
    expert_filenames = [f'{CFG.model}_{CFG.dataset.lower()}_{src.lower()}_meta_{CFG.prompt_size}.pt' for src in src_domains]
elif CFG.use_zero:
    expert_filenames = [f'{CFG.model}_{CFG.dataset.lower()}_{src.lower()}_zero_{CFG.prompt_size}.pt' for src in src_domains]
elif CFG.use_unif:
    expert_filenames = [f'{CFG.model}_{CFG.dataset.lower()}_{src.lower()}_unif_{CFG.prompt_size}.pt' for src in src_domains]
elif CFG.use_rand:
    expert_filenames = None
else:
    expert_filenames = [f'{CFG.model}_{CFG.dataset.lower()}_{src.lower()}_{CFG.prompt_size}.pt' for src in src_domains]

if expert_filenames is None:
    expert_state_dicts = [Prompter(224, CFG.prompt_size).state_dict() for _ in src_domains]
else:
    expert_state_dicts = [torch.load(expert_dir.joinpath(fn))['best_state_dict'] for fn in expert_filenames]

experts = []
repeat = 1 if CFG.aug_pt is None else CFG.aug_pt
jitter = CFG.aug_pt is not None
with torch.no_grad():
    for _ in range(repeat):
        for state_dict in expert_state_dicts:
            expert = Prompter(224, CFG.prompt_size).cuda(DEVICE)
            expert.load_state_dict(state_dict)
            if jitter:
                expert.jitter_normal(std=0.03)
            experts.append(expert)

if CFG.num_experts is not None:
    experts = experts[:CFG.num_experts]
    
prompter = PromptPool(prompters=experts, embed_dim=512, normalize_experts=CFG.norm_experts, 
                      use_softmax=CFG.use_softmax, use_tanh=CFG.use_tanh,
                      uniform_mix=CFG.mix_uniform, random_mix=CFG.mix_random).cuda(DEVICE)
if CFG.model == 'resnet50.clip':
    net: nn.Module = clip.load('RN50', device='cpu')[0].visual.cuda(DEVICE)
    head: nn.Module = net.attnpool.c_proj
elif CFG.model == 'vit_base_patch16.tv_in1k':
    net: nn.Module = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).cuda(DEVICE)
    head: nn.Module = net.heads.head
else:
    net: nn.Module = timm.create_model(CFG.model, pretrained=True).cuda(DEVICE)
    head: nn.Module = net.get_classifier()
head.weight = nn.Parameter(head.weight[:len(classes)])
if head.bias is not None:
    head.bias = nn.Parameter(head.bias[:len(classes)])
head.out_features = len(classes)

def clone_state_dicts():
    net_state_dict = {key: val.clone().cpu() for key, val in net.state_dict().items()}
    prompter_state_dict = {key: val.clone().cpu() for key, val in prompter.state_dict().items()}
    return net_state_dict, prompter_state_dict

miner_fn = pml_miners.TripletMarginMiner().cuda(DEVICE)
ml_loss_fn = pml_losses.TripletMarginLoss().cuda(DEVICE)
if CFG.use_kld:
    dist_loss_fn = nn.KLDivLoss(reduction='batchmean').cuda(DEVICE)
else:
    dist_loss_fn = nn.CrossEntropyLoss().cuda(DEVICE)

def negvar_loss_fn(embeddings, targets):
    target_candidates = torch.unique(targets)
    loss = 0
    for tar in target_candidates:
        tar_indices = (targets == tar).nonzero().flatten()
        tar_embeddings = embeddings[tar_indices]
        loss -= torch.var(tar_embeddings, dim=0).mean()
    return loss

parameters = list(prompter.parameters(recurse=CFG.tune_more)) + list(head.parameters()) 
optimizer = optim.AdamW(parameters, lr=CFG.lr, betas=(0.5, 0.999))
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=CFG.epoch, eta_min=CFG.lr * 0.1)

def save_sample_image(filename):
    zero = torch.zeros(1, 3, 224, 224).cuda(DEVICE)
    left = prompter(zero)
    right = tar_dataset[0][0][None].cuda(DEVICE)
    center = prompter(right)
    
    left, center, right = left[0].cpu(), center[0].cpu(), right[0].cpu()
    center = center * std + mean
    right = right * std + mean
    
    sample = make_grid([left, center, right])
    save_image(sample, filename)
    
save_sample_image(INIT_SAMPLE_FILENAME)

# TRAINING LOOP
if not CFG.profile:
    loss, match, total = 0, 0, 0
    net.eval()
    prompter.eval()
    with tqdm(valid_loader, desc='INIT', position=1, leave=False, dynamic_ncols=True) as valid_bar, torch.no_grad():
        for inputs, targets in valid_bar:
            batch_size = inputs.size(0)
            inputs, targets = inputs.cuda(DEVICE), targets.cuda(DEVICE)
            onehots = one_hot(targets, len(classes)).float().cuda(DEVICE)
            
            prompted_inputs = prompter(inputs)
            # feats = net.forward_features(prompted_inputs)
            # embedding = torch.flatten(feats, start_dim=1)
            # outs = net.forward_head(feats)
            # outs = torch.softmax(outs, dim=1)
            outs = torch.softmax(net(prompted_inputs), dim=1)
            
            # ml_pairs = miner_fn(embedding, targets)
            # ml_loss = ml_loss_fn(embedding, targets, ml_pairs)
            # ce_loss = dist_loss_fn(outs, onehots)
            # batch_loss = ml_loss * CFG.ml_loss + ce_loss
            batch_loss = dist_loss_fn(outs, onehots)
            
            loss += (batch_loss * batch_size).item()
            match += (targets == outs.argmax(dim=1)).sum().item()
            total += batch_size
            
            valid_loss = loss / total
            valid_accuracy = match / total
            valid_bar.set_postfix_str(f'loss={valid_loss:.3f} | top-1={valid_accuracy * 100:.2f}%')
    print(f'Init: loss={valid_loss:.3f} | top-1={valid_accuracy * 100:.2f}%')

from time import time
training_time = 0

with tqdm(range(1, CFG.epoch + 1), desc='EPOCH', position=1, leave=False, dynamic_ncols=True) as epoch_bar:
    best_epoch, best_accuracy, best_state_dict = -1, -1, None
    lr_list, loss_list, accuracy_list = [], [], []
    
    for epoch in epoch_bar:
        if CFG.profile and epoch == 1:
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True)
            prof.__enter__()
        else:
            prof = None
            
        with tqdm(train_loader, desc='TRAIN', position=2, leave=False, dynamic_ncols=True) as train_bar:
            loss, match, total = 0, 0, 0
            
            net.train()
            prompter.train()
            for inputs, targets in train_bar:
                batch_size = inputs.size(0)
                inputs, targets = inputs.cuda(DEVICE), targets.cuda(DEVICE)
                onehots = one_hot(targets, len(classes)).float().cuda(DEVICE)
                
                start = time()
                prompted_inputs = prompter(inputs)
                
                outs = torch.softmax(net(prompted_inputs), dim=1)
                batch_loss = dist_loss_fn(outs, onehots)
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                training_time += time() - start
                
                loss += (batch_loss * batch_size).item()
                match += (targets == outs.argmax(dim=1)).sum().item()
                total += batch_size
                
                train_loss = loss / total
                train_accuracy = match / total
                train_bar.set_postfix_str(f'loss={train_loss:.3f} | top-1={train_accuracy * 100:.2f}%')
            scheduler.step(epoch)
            lr_list.append(optimizer.param_groups[0]['lr'])
            loss_list.append(train_loss)
            
        if CFG.profile:
            prof.__exit__(None, None, None)
            torch.save(prof.key_averages(), 'temp/temp.pt')
            print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            exit()
            
        with tqdm(valid_loader, desc='VALID', position=2, leave=False, dynamic_ncols=True) as valid_bar, torch.no_grad():
            loss, match, total = 0, 0, 0
            
            net.eval()
            prompter.eval()
            for inputs, targets in valid_bar:
                batch_size = inputs.size(0)
                inputs, targets = inputs.cuda(DEVICE), targets.cuda(DEVICE)
                onehots = one_hot(targets, len(classes)).float().cuda(DEVICE)
                
                prompted_inputs = prompter(inputs)
                # feats = net.forward_features(prompted_inputs)
                # embedding = torch.flatten(feats, start_dim=1)
                # outs = net.forward_head(feats)
                # outs = torch.softmax(outs, dim=1)
                outs = torch.softmax(net(prompted_inputs), dim=1)
                
                # ml_pairs = miner_fn(embedding, targets)
                # ml_loss = ml_loss_fn(embedding, targets, ml_pairs)
                # ce_loss = dist_loss_fn(outs, onehots)
                # batch_loss = ml_loss * CFG.ml_loss + ce_loss
                batch_loss = dist_loss_fn(outs, onehots)
                
                loss += (batch_loss * batch_size).item()
                match += (targets == outs.argmax(dim=1)).sum().item()
                total += batch_size
                
                valid_loss = loss / total
                valid_accuracy = match / total
                valid_bar.set_postfix_str(f'loss={valid_loss:.3f} | top-1={valid_accuracy * 100:.2f}%')
            accuracy_list.append(valid_accuracy)
                
        if valid_accuracy >= best_accuracy:
            best_accuracy = valid_accuracy
            best_epoch = epoch
            best_state_dict = clone_state_dicts()
            save_sample_image(BEST_SAMPLE_FILENAME)
            
            epoch_bar.set_postfix_str(f'loss={valid_loss:.3f} | top-1={valid_accuracy * 100:.2f}% @ {epoch}')
        
        save_sample_image(LAST_SAMPLE_FILENAME)
        writer.add_scalar('base/learning rate',  lr_list[-1],        epoch)
        writer.add_scalar('base/loss',           loss_list[-1],      epoch)
        writer.add_scalar('base/accuracy',       accuracy_list[-1],  epoch)
        torch.save({
            'cfg': CFG,
            'epoch': best_epoch,
            'accuracy': best_accuracy,
            'lr_list': lr_list,
            'loss_list': loss_list,
            'accuracy_list': accuracy_list,
            'best_state_dict': best_state_dict,
            'last_state_dict': clone_state_dicts(),
        }, CKPT_FILENAME)
        
        if CFG.early_stop and (best_accuracy * 100.0 >= CFG.early_stop):
            print(f'Early Stop at {best_epoch} epoch, {best_accuracy * 100:.2f}%')
            break

print(f'{CFG.dataset} dataset took {training_time / CFG.epoch} seconds per epoch for training.')
