import torch
import torch.nn as nn

import sys
sys.path.append("..")

from model import encoder
from thirdparty import siren

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, embed_frq=16, output_ch=1, skips=[4]):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = 3 + 6 * embed_frq
        self.skips = skips
        
        self.embedding = encoder.Embedding(embed_frq)
        self.layers = nn.ModuleList()
        
        for i in range(D):
            if i == 0:
                layer = nn.Linear(self.input_ch, W)
            elif i in skips:
                layer = nn.Linear(W+self.input_ch, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU())
            self.layers.append(layer)
            
        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        x = self.embedding(x)
        h = x
        for i in range(len(self.layers)):
            if i in self.skips:
                h = torch.cat([x, h], -1)
            h = self.layers[i](h)

        outputs = self.output_linear(h)
        outputs = torch.sigmoid(outputs)
        
        return outputs
    
    def gradient(self, p):
        with torch.enable_grad():
            p.requires_grad_(True)
            y = self.forward(p)[...,:1]
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=p,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True, allow_unused=True)[0]
            return gradients.unsqueeze(1)

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(model.weight, 0.0, 0.02)

class occNeRF(nn.Module):
    def __init__(self, D=2, W=64, output_ch=1, x_enc_ch=32, skips=[1],
                 bbox = ([-15.0, -15.0, -15.0], [15.0, 15.0, 15.0])):
        super(occNeRF, self).__init__()
        # 深度、宽度、32 + 编码层通道数、跳跃层
        self.D = D
        self.W = W
        self.input_ch = 32 + x_enc_ch
        self.skips = skips
        # 场景的边界框用于 hash 编码
        bbox = (torch.tensor(bbox[0], dtype = torch.float32).cuda(), \
                torch.tensor(bbox[1], dtype = torch.float32).cuda())
        # 对 bbox 场景进行多分辨率 hash 编码
        self.embedding = encoder.HashEmbedder(bbox)
        self.x_concat_layer = nn.Linear(3, x_enc_ch)
        self.layers = nn.ModuleList()
        
        # 定义每一层的输入输出
        for i in range(D):
            if i == 0:
                layer = nn.Linear(self.input_ch, W)
            elif i in skips:
                layer = nn.Linear(W+self.input_ch, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, siren.Siren(W, W))
            self.layers.append(layer)
            
        # 输出网格是否被占据，故输出为一维
        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        # hash 编码信息
        x_hash = self.embedding(x)
        # 原始位置信息经过线性层
        x_enc = self.x_concat_layer(x)
        # 合并两者信息
        x = torch.cat([x_hash, x_enc], -1)
        h = x
        # 前馈，参考原文 Fig.4
        for i in range(len(self.layers)):
            if i in self.skips:
                h = torch.cat([x, h], -1)
            h = self.layers[i](h)

        outputs = self.output_linear(h)
        outputs = torch.sigmoid(outputs)
        
        return outputs
    
    def gradient(self, p):
        with torch.enable_grad():
            p.requires_grad_(True)
            y = self.forward(p)[...,:1]
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=p,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True, allow_unused=True)[0]
            return gradients.unsqueeze(1)
