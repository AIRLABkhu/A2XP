import torch
from torch import nn
from models.modules import Prompter
import timm
import clip


class Classifier(nn.Module):
    def __init__(self, name: str, num_classes: int, use_head: bool=False,
                 prompt_size: int=30, prompt_init: float=0.3, bias: bool=True):
        super(Classifier, self).__init__()
        
        if name == 'resnet50.clip':
            self.body = clip.load('RN50', device='cpu')
            body_out_dim = 1024
            image_size = 224
        else:
            self.body = timm.create_model(name, pretrained=True)
            body_out_dim = self.body.default_cfg['num_classes']
            image_size = self.body.default_cfg['input_size'][2]
        self.prompter = Prompter(image_size, prompt_size, prompt_init)
        
        if use_head:
            self.__bias = bias
            self.head = nn.Linear(body_out_dim, num_classes, bias=bias)
        else:
            self.__bias = False
            self.head = nn.Identity()
        
    def adapters(self):
        if self.prompter.has_values:
            yield self.prompter.prompt_t
            yield self.prompter.prompt_b
            yield self.prompter.prompt_l
            yield self.prompter.prompt_r
        yield self.head.weight
        if self.__bias:
            yield self.head.bias
        
    def forward_feats(self, x):
        x = self.prompter(x)
        return self.body.forward_features(x)
    
    def forward_logits(self, feats):
        return self.body.forward_head(feats)
    
    def forward_head(self, logits):
        return self.head(logits)
    
    def forward(self, x):
        x = self.prompter(x)
        logits = self.body(x)
        return self.head(logits)
