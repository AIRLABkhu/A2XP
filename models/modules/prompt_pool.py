if __name__ == '__main__':
    import os, sys
    __root = __file__
    for _ in range(3):
        __root = os.path.dirname(__root)
    sys.path.append(__root)

from typing import Iterator
from typing import Iterable

import torch
from torch import nn
from torch.nn.parameter import Parameter

from models.modules import Prompter
import timm


class PromptPool(nn.Module):
    def __init__(self, prompters: Iterable[Prompter], embed_dim: int):
        super(PromptPool, self).__init__()
        
        self._num_prompts = len(prompters)
        
        self.encoder = timm.create_model('resnet18', pretrained=True, num_classes=embed_dim)
        self.keys = nn.Parameter(torch.randn(embed_dim, self._num_prompts)) 
        self.prompters = nn.ModuleList(prompters)
        
    def parameters(self, recurse: bool=True) -> Iterator[Parameter]:
        yield self.keys
        if recurse:
            yield self.encoder.get_classifier().weight
            if self.encoder.get_classifier().bias is not None:
                yield self.encoder.get_classifier().bias
            # for params in self.prompters.parameters():
            #     yield params
                
    def forward(self, x):  # ...............................| batch_size, C, H, W
        if self._num_prompts == 0:
            return x
        
        with torch.no_grad():
            x_feats = self.encoder.forward_features(x)
        x_embed = self.encoder.forward_head(x_feats)  # ....| batch_size, embed_dim
        
        attn = torch.mm(x_embed, self.keys)  # .............| batch_size, num_prompts
        attn = attn[:, :, None, None, None]  # .............| batch_size, num_prompts, 1, 1, 1
        
        prompts = [prompter.prompt for prompter in self.prompters]
        prompts = torch.cat(prompts)  # ....................| num_prompts, C, H, W
        prompts = prompts.unsqueeze(0)  # ..................| 1, num_prompts, C, H, W
        
        attn_prompts = torch.multiply(attn, prompts)  # ....| batch_size, num_prompts, C, H, W
        attn_prompts = torch.sum(attn_prompts, dim=1)  # ...| batch_size, C, H, W
        
        return torch.add(x, attn_prompts)


class ProbabilisticPromptPool(nn.Module):
    def __init__(self, prompters: Iterable[Prompter], embed_dim: int, backbone: str='resnet18',
                 normalize_experts: bool=False, use_softmax: bool=False, use_tanh: bool=False,
                 uniform_mix: bool=False, random_mix: bool=False):
        super(ProbabilisticPromptPool, self).__init__()
        
        assert not (uniform_mix and random_mix)
        self.uniform_mix = uniform_mix
        self.random_mix = random_mix
        
        self._num_prompts = len(prompters)
        
        self.shared_encoder = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.inputs_head = nn.Linear(self.shared_encoder.num_features, embed_dim)
        self.prompt_head = nn.Linear(self.shared_encoder.num_features, embed_dim)
        self.prompters = nn.ModuleList(prompters)
        
        if normalize_experts:
            with torch.no_grad():
                for prompter in self.prompters:
                    prompter: Prompter
                    prompt_norm = torch.norm(torch.cat([
                        prompter.prompt_t,
                        prompter.prompt_b,
                        prompter.prompt_l.transpose(2, 3),
                        prompter.prompt_r.transpose(2, 3),
                    ], dim=3), p=2)
                    prompter.prompt_t /= prompt_norm
                    prompter.prompt_b /= prompt_norm
                    prompter.prompt_l /= prompt_norm
                    prompter.prompt_r /= prompt_norm
        self.use_softmax = use_softmax
        self.use_tanh = use_tanh
        
        self.__retain_attn_weights = False
        self.__attn_weights = None
        
    @property
    def retain_attn_weights(self):
        return self.__retain_attn_weights
    
    @property
    def attn_weights(self):
        return self.__attn_weights
        
    def retain_attn_weights_(self, value: bool):
        self.__retain_attn_weights = value
        
    def parameters(self, recurse: bool=False) -> Iterator[Parameter]:
        yield self.inputs_head.weight
        yield self.inputs_head.bias
        yield self.prompt_head.weight
        yield self.prompt_head.bias
        if recurse:
            for param in self.prompters.parameters():
                yield param
                
    def forward(self, x):  # ...............................| batch_size, C, H, W
        if self._num_prompts == 0:
            return x
        batch_size = x.size(0)
        
        prompts = [prompter.prompt for prompter in self.prompters]
        prompts = torch.cat(prompts)
        
        if self.uniform_mix:
            attn = torch.ones(
                batch_size, self._num_prompts, 1, 1, 1, 
                dtype=prompts.dtype, device=prompts.device
            ) / self._num_prompts
        elif self.random_mix:
            attn = torch.rand(
                batch_size, self._num_prompts, 1, 1, 1, 
                dtype=prompts.dtype, device=prompts.device
            )
            attn /= attn.sum(dim=1, keepdim=True)
        else:
            x_prompts = torch.cat([x, prompts])
            with torch.no_grad():
                feats = self.shared_encoder(x_prompts)
            inputs_feats, prompt_feats = torch.split(feats, [batch_size, self._num_prompts])
            
            inputs_embed = self.inputs_head(inputs_feats)  # ...| batch_size, embed_dim
            prompt_embed = self.prompt_head(prompt_feats)  # ...| num_prompts, embed_dim
            
            attn = torch.mm(inputs_embed, prompt_embed.T)  # ...| batch_size, num_prompts
            if self.use_softmax:
                attn = torch.softmax(attn, dim=1)
            if self.use_tanh:
                attn = torch.tanh(attn)
            if self.__retain_attn_weights:
                self.__attn_weights = attn.clone().detach().cpu()
            attn = attn[:, :, None, None, None]  # .............| batch_size, num_prompts, 1, 1, 1
        
        prompts = prompts.unsqueeze(0)  # ..................| 1, num_prompts, C, H, W
        attn_prompts = torch.multiply(attn, prompts)  # ....| batch_size, num_prompts, C, H, W
        attn_prompts = torch.sum(attn_prompts, dim=1)  # ...| batch_size, C, H, W
        
        return torch.add(x, attn_prompts)
