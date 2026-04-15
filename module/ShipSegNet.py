

import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.instancenorm import InstanceNorm2d
from torch.nn import LeakyReLU
import torch.nn.functional as F

class MultiScaleRFBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch1 = nn.Conv2d(channels, channels, 1)
        self.branch2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.branch3 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.fuse = nn.Conv2d(channels * 3, channels, 1)
        self.norm = InstanceNorm2d(channels, affine=True)
        self.act = LeakyReLU(inplace=True)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.fuse(out)
        out = self.norm(out)
        out = self.act(out)
        return x + out   

class BackgroundAwareSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.bias = nn.Parameter(torch.full((1, channels), -2.0))  

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc(y) + self.bias   
        y = torch.sigmoid(y).view(b, c, 1, 1)
        return x * y


class ConvNormReLU(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride):
        super().__init__()

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True
        )


        self.norm = InstanceNorm2d(out_channels, eps=1e-05, affine=True)


        self.relu = LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.features_per_stage = [32, 64, 128, 256, 512, 512, 512, 512]

        self.strides = [
            [1, 1], [2, 2], [2, 2], [2, 2],
            [2, 2], [2, 2], [2, 2], [2, 2]
        ]

        stages = []
        in_channels = 3  
        for i in range(len(self.strides)):

            stage = nn.Sequential(
                ConvNormReLU(in_channels, self.features_per_stage[i], self.strides[i]),
                ConvNormReLU(self.features_per_stage[i], self.features_per_stage[i], 1),
                MultiScaleRFBlock(self.features_per_stage[i]),
                BackgroundAwareSE(self.features_per_stage[i]) 
            )
            stages.append(stage)
            in_channels = self.features_per_stage[i]  

        self.stages = nn.Sequential(*stages)

    def forward(self, x):
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


class MSRM(nn.Module):
    def __init__(self, channels=2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.fuse = nn.Conv2d(channels*2, channels, 1)
        self.norm = nn.InstanceNorm2d(channels, affine=True)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out1 = self.act(self.conv1(x))
        out2 = self.act(self.conv2(x))
        out = torch.cat([out1, out2], dim=1)
        out = self.fuse(out)
        out = self.norm(out)
        out = self.act(out)
        return x + out   


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h], indexing='xy')).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.reshape(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing='xy')
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale).reshape(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

class Decoder_Dy(nn.Module):

    def __init__(self):
        super().__init__()

        self.features_per_stage = [32, 64, 128, 256, 512, 512, 512, 512]

        self.strides = [
            [1, 1], [2, 2], [2, 2], [2, 2],
            [2, 2], [2, 2], [2, 2], [2, 2]
        ]

        n_encoder_stages = len(self.features_per_stage)  # 8
        n_decoder_stages = n_encoder_stages - 1          # 7

        self.dysamples = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.seg_layers = nn.ModuleList()

        for s in range(1, n_encoder_stages):
            input_features_below = self.features_per_stage[-s]      
            input_features_skip = self.features_per_stage[-(s + 1)] 


            concat_channels = input_features_below + input_features_skip


            self.dysamples.append(
                DySample(input_features_below, scale=2, style='lp', groups=4, dyscope=False)
            )


            self.stages.append(
                nn.Sequential(
                    ConvNormReLU(concat_channels, input_features_skip, 1),
                    ConvNormReLU(input_features_skip, input_features_skip, 1)
                )
            )


            self.seg_layers.append(
                Conv2d(input_features_skip, 2, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, skips):

        lres_input = skips[-1]
        seg_outputs = []

        for s in range(len(self.stages)):

            x = self.dysamples[s](lres_input)


            skip = skips[-(s + 2)]


            x = torch.cat((x, skip), dim=1)


            x = self.stages[s](x)


            if s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))

            lres_input = x


        seg_outputs = seg_outputs[::-1]
        return seg_outputs[0]

class ShipSegNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder_Dy()
        self.msrm = MSRM()

    def forward(self, x):
        skips = self.encoder(x)
        out = self.decoder(skips)
        out = self.msrm(out) 
        return out



if __name__ == '__main__':
    model = ShipSegNet()
    dummy_input = torch.randn(1, 3, 512, 512)
    output = model(dummy_input)
    print(f"{dummy_input.shape}")
    print(f"{output.shape}")