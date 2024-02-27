if __name__ == '__main__':
    import os, sys
    __root = __file__
    for _ in range(3):
        __root = os.path.dirname(__root)
    sys.path.append(__root)

from typing import Iterator
import torch
from torch import nn
from torch.nn.parameter import Parameter

from models.modules import Prompter, PromptPool
import timm


class AttentivePrompter(PromptPool):
    def __init__(self, image_size: int, prompt_size: int, init: float, 
                 num_prompts: int, embed_dim: int):
        prompters = [Prompter(image_size, prompt_size, init) for _ in range(num_prompts)]
        super(AttentivePrompter, self).__init__(prompters, embed_dim)
        
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        yield self.keys
        if recurse:
            yield self.encoder.get_classifier().weight
            if self.encoder.get_classifier().bias is not None:
                yield self.encoder.get_classifier().bias
            for params in self.prompters.parameters():
                yield params
                
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
    

if __name__ == '__main__':
    prompter = AttentivePrompter(224, 30, 0.03, 4, 512)
    sample = torch.randn(2, 3, 224, 224)
    output = prompter(sample)
    
    input_shape = sample.shape
    output_shape = output.shape
    
    assert input_shape == output_shape
    print('SHAPE_TEST_PASSED', __file__)
    
    loss = output.sum()
    loss.backward()
    print('BACKWARD_TEST_PASSED', __file__)
