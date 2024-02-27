import torch
from torch import nn


class Prompter(nn.Module):
    def __init__(self, image_size: int, prompt_size: int, init: float=0.3):
        super(Prompter, self).__init__()
        
        self.__prompt_size = prompt_size
        self.__init = init
        
        prompt_size_v = (prompt_size, image_size)
        prompt_size_h = (image_size - prompt_size * 2, prompt_size)
        self.base_size = image_size - prompt_size * 2
        self.prompt_t = nn.Parameter(torch.randn(1, 3, *prompt_size_v) * init)
        self.prompt_b = nn.Parameter(torch.randn(1, 3, *prompt_size_v) * init)
        self.prompt_l = nn.Parameter(torch.randn(1, 3, *prompt_size_h) * init)
        self.prompt_r = nn.Parameter(torch.randn(1, 3, *prompt_size_h) * init)
        
    @property
    def has_values(self):
        return (self.__prompt_size != 0) and (self.__init != 0)
        
    @property
    def prompt(self):
        result = torch.zeros(1, 3, self.base_size, self.base_size).to(self.prompt_l.device)
        result = torch.cat([self.prompt_l, result, self.prompt_r], dim=3)
        result = torch.cat([self.prompt_t, result, self.prompt_b], dim=2)
        return result
    
    @torch.no_grad()
    def jitter_normal(self, std: float, mean: float=0.0):
        self.prompt_t += torch.randn_like(self.prompt_t) * std + mean
        self.prompt_b += torch.randn_like(self.prompt_b) * std + mean
        self.prompt_l += torch.randn_like(self.prompt_l) * std + mean
        self.prompt_r += torch.randn_like(self.prompt_r) * std + mean
    
    def forward(self, x):
        return x + self.prompt
