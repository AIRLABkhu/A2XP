from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path

import random
import numpy as np

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.nn.functional import one_hot, cross_entropy
from torch.utils.data import DataLoader
from torch.backends import cudnn

import timm, clip
from torchvision.models import vit_b_16, ViT_B_16_Weights
from models.modules import Prompter
from domainbed import get_dataset, get_domains_and_classes


parser = ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--seed', default=None, type=int)

parser.add_argument('--model', default='vit_base_patch16_clip_224.openai')
parser.add_argument('--init-param', type=float, default=0.03)
parser.add_argument('--constant-init', type=float, default=None)
parser.add_argument('--meta-init', type=str, default=None)
parser.add_argument('--zero-init', action='store_true', default=False)
parser.add_argument('--unif-init', action='store_true', default=False)
parser.add_argument('--prompt-size', type=int, default=30)

parser.add_argument('--dataset', default='pacs')
parser.add_argument('--domain', default='p')
parser.add_argument('--make-anchor', action='store_true', default=False)
parser.add_argument('--use-anchor', action='store_true', default=False)
parser.add_argument('--batch-size', type=int, default=128)

parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--lr', type=float, default=1.0E+4)

parser.add_argument('--early-stop', type=float, default=None)
parser.add_argument('--manual-tag', type=str, default=None)

CFG = parser.parse_args()
DEVICE = CFG.gpu

SEED = torch.initial_seed() if CFG.seed is None else CFG.seed

random.seed(SEED)
np.random.seed(SEED)
cudnn.deterministic = CFG.seed is not None
cudnn.benchmark = not cudnn.deterministic
torch.cuda.set_device(DEVICE)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

if CFG.make_anchor and CFG.use_anchor:
    raise ValueError("'make-anchor' and 'use-anchor' are mutually exclusive.")

if (CFG.meta_init is not None) + CFG.zero_init + CFG.unif_init > 1:
    raise ValueError("'meta-init', 'zero-init' and 'unif-init' are mutually exclusive.")

EXP_DIR = Path('./log').joinpath('experts')
EXP_DIR.mkdir(parents=True, exist_ok=True)
if CFG.manual_tag is not None:
    CKPT_FILENAME = EXP_DIR.joinpath(f'{CFG.manual_tag}.pt')
elif CFG.make_anchor:
    CKPT_FILENAME = EXP_DIR.joinpath(f'{CFG.model}_{CFG.dataset.lower()}_{CFG.domain.lower()}_anch_{CFG.prompt_size}.pt')
elif CFG.use_anchor:
    CKPT_FILENAME = EXP_DIR.joinpath(f'{CFG.model}_{CFG.dataset.lower()}_{CFG.domain.lower()}_vect_{CFG.prompt_size}.pt')
    
elif CFG.meta_init is not None:
    CKPT_FILENAME = EXP_DIR.joinpath(f'{CFG.model}_{CFG.dataset.lower()}_{CFG.domain.lower()}_meta_{CFG.prompt_size}.pt')
elif CFG.zero_init:
    CKPT_FILENAME = EXP_DIR.joinpath(f'{CFG.model}_{CFG.dataset.lower()}_{CFG.domain.lower()}_zero_{CFG.prompt_size}.pt')
elif CFG.unif_init:
    CKPT_FILENAME = EXP_DIR.joinpath(f'{CFG.model}_{CFG.dataset.lower()}_{CFG.domain.lower()}_unif_{CFG.prompt_size}.pt')
    
else:
    CKPT_FILENAME = EXP_DIR.joinpath(f'{CFG.model}_{CFG.dataset.lower()}_{CFG.domain.lower()}_{CFG.prompt_size}.pt')
print(f'{CKPT_FILENAME=}')
    
META_DIR = EXP_DIR.joinpath('meta')
META_INIT_FILENAME = META_DIR.joinpath(f'{CFG.model}_{CFG.meta_init}.pt')

domains, classes = get_domains_and_classes(CFG.dataset)
src_dataset, tar_dataset = get_dataset(CFG.dataset, target=CFG.domain)
if CFG.make_anchor:
    tar_dataset = src_dataset

def seed_worker(worker_id):
    np.random.seed(SEED)
    random.seed(SEED)

g = torch.Generator()
g.manual_seed(0)

loader_kwargs = dict(
    # dataset=tar_dataset,
    batch_size=CFG.batch_size,
    num_workers=8,
    worker_init_fn=seed_worker,
    generator=g,
)
train_loader = DataLoader(tar_dataset, shuffle=True, **loader_kwargs)
valid_loader = DataLoader(src_dataset, shuffle=False, **loader_kwargs)

if CFG.use_anchor: 
    anch_filename = EXP_DIR.joinpath(f'{CFG.model}_{CFG.dataset.lower()}_{CFG.domain.lower()}_anch_{CFG.prompt_size}.pt')
    anchor_state_dict = torch.load(anch_filename)['best_state_dict']
    anchor = Prompter(224, CFG.prompt_size, init=CFG.init_param)
    anchor.load_state_dict(anchor_state_dict)
    anchor = anchor.prompt.clone().detach().cuda(DEVICE)
else:
    anchor = 0.0

prompter = Prompter(224, CFG.prompt_size, init=CFG.init_param).cuda(DEVICE)
with torch.no_grad():
    if CFG.meta_init is not None:
        prompter_state_dict = torch.load(META_INIT_FILENAME, map_location='cpu')
        prompter.load_state_dict(prompter_state_dict)
    elif CFG.zero_init:
        prompter.prompt_t = nn.Parameter(torch.zeros_like(prompter.prompt_t))
        prompter.prompt_b = nn.Parameter(torch.zeros_like(prompter.prompt_b))
        prompter.prompt_l = nn.Parameter(torch.zeros_like(prompter.prompt_l))
        prompter.prompt_r = nn.Parameter(torch.zeros_like(prompter.prompt_r))
    elif CFG.unif_init:
        prompter.prompt_t = nn.Parameter((torch.rand_like(prompter.prompt_t) * 2 - 1) * CFG.init_param)
        prompter.prompt_b = nn.Parameter((torch.rand_like(prompter.prompt_b) * 2 - 1) * CFG.init_param)
        prompter.prompt_l = nn.Parameter((torch.rand_like(prompter.prompt_l) * 2 - 1) * CFG.init_param)
        prompter.prompt_r = nn.Parameter((torch.rand_like(prompter.prompt_r) * 2 - 1) * CFG.init_param)
    elif CFG.constant_init:
        prompter.prompt_t = nn.Parameter(torch.ones_like(prompter.prompt_t) * CFG.constant_init)
        prompter.prompt_b = nn.Parameter(torch.ones_like(prompter.prompt_b) * CFG.constant_init)
        prompter.prompt_l = nn.Parameter(torch.ones_like(prompter.prompt_l) * CFG.constant_init)
        prompter.prompt_r = nn.Parameter(torch.ones_like(prompter.prompt_r) * CFG.constant_init)
    else:
        prompter.prompt_t = nn.Parameter(torch.randn_like(prompter.prompt_t) * CFG.init_param)
        prompter.prompt_b = nn.Parameter(torch.randn_like(prompter.prompt_b) * CFG.init_param)
        prompter.prompt_l = nn.Parameter(torch.randn_like(prompter.prompt_l) * CFG.init_param)
        prompter.prompt_r = nn.Parameter(torch.randn_like(prompter.prompt_r) * CFG.init_param)

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

optimizer = optim.SGD(prompter.parameters(), lr=CFG.lr, momentum=0.9)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150, 180, 210], gamma=0.1)

with tqdm(range(1, CFG.epoch + 1), desc='EPOCH', position=1, leave=False, dynamic_ncols=True) as epoch_bar:
    best_epoch, best_accuracy, best_state_dict = -1, -1, None
    lr_list, loss_list, accuracy_list = [], [], []
    
    for epoch in epoch_bar:
        with tqdm(train_loader, desc='TRAIN', position=2, leave=False, dynamic_ncols=True) as train_bar:
            loss, match, total = 0, 0, 0
            
            net.train()
            prompter.train()
            for inputs, targets in train_bar:
                batch_size = inputs.size(0)
                inputs, targets = inputs.cuda(DEVICE), targets.cuda(DEVICE)
                onehots = one_hot(targets, len(classes)).float().cuda(DEVICE)
                
                prompted_inputs = prompter(inputs) + anchor
                outs = net(prompted_inputs)
                batch_loss = cross_entropy(outs, onehots)
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                loss += (batch_loss * batch_size).item()
                match += (targets == outs.argmax(dim=1)).sum().item()
                total += batch_size
                
                train_loss = loss / total
                train_accuracy = match / total
                train_bar.set_postfix_str(f'loss={train_loss:.3f} | top-1={train_accuracy * 100:.2f}%')
            scheduler.step(epoch)
            lr_list.append(optimizer.param_groups[0]['lr'])
            loss_list.append(train_loss)
            
        with tqdm(valid_loader, desc='VALID', position=2, leave=False, dynamic_ncols=True) as valid_bar, torch.no_grad():
            loss, match, total = 0, 0, 0
            
            net.eval()
            prompter.eval()
            for inputs, targets in valid_bar:
                batch_size = inputs.size(0)
                inputs, targets = inputs.cuda(DEVICE), targets.cuda(DEVICE)
                onehots = one_hot(targets, len(classes)).float().cuda(DEVICE)
                
                prompted_inputs = prompter(inputs) + anchor
                outs = net(prompted_inputs)
                batch_loss = cross_entropy(outs, onehots)
                
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
            best_state_dict = {key: val.clone().cpu() for key, val in prompter.state_dict().items()}
            
            epoch_bar.set_postfix_str(f'loss={valid_loss:.3f} | top-1={valid_accuracy * 100:.2f}% @ {epoch}')
        
        torch.save({
            'cfg': CFG,
            'epoch': best_epoch,
            'accuracy': best_accuracy,
            'lr_list': lr_list,
            'loss_list': loss_list,
            'accuracy_list': accuracy_list,
            'best_state_dict': best_state_dict,
            'last_state_dict': {key: val.clone().cpu() for key, val in prompter.state_dict().items()},
        }, CKPT_FILENAME)
        
        if CFG.early_stop and (best_accuracy * 100.0 >= CFG.early_stop):
            print(f'Early Stop at {best_epoch} epoch, {best_accuracy * 100:.2f}%')
            break

print(f'Best: loss={loss_list[best_epoch - 1]:.3f} | top-1={best_accuracy * 100:.2f}% @ {best_epoch}')
