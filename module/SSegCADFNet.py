import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


import xformers.ops as xops

class XFormersAttention(nn.Module):
    def __init__(self, channels, num_heads=2):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.conv = nn.Conv2d(channels, channels, 3, 1, padding="same", groups=channels)
        self.layernorm = Layernorm(channels, eps=1e-5)
        
        # 线性投影层
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        
        self.att_value = nn.Sequential(
            self.conv,
            nn.ReLU(),
            self.layernorm            
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 准备输入
        att_value = self.att_value(x)
        att_value = att_value.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        
        # 生成QKV
        q = self.q_proj(att_value)
        k = self.k_proj(att_value)
        v = self.v_proj(att_value)
        
        # 使用xFormers的内存高效注意力
        output = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=None,
            op=None  # 自动选择最佳实现
        )
        
        # 输出投影
        output = self.out_proj(output)
        
        # 恢复形状
        output = output.permute(0, 2, 1).view(B, C, H, W)
        return output

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

class BottConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x

class Layernorm(nn.Module):
    def __init__(self,channels,eps):
        super().__init__()

        self.layernorm = nn.LayerNorm(channels,eps=eps)

    def forward(self,x):
        x = x.permute(0,2,3,1)
        x = self.layernorm(x)
        x = x.permute(0,3,1,2)

        return x

class Attention(nn.Module):
    def __init__(self,
                 channels,
                 ):
        super().__init__()
        
        self.conv = nn.Conv2d(channels, channels, 3, 1, padding="same", bias=True, groups=channels)
        self.layernorm = Layernorm(channels, eps=1e-5)

        self.att_value = nn.Sequential(
            self.conv,
            nn.ReLU(),
            self.layernorm            
        )
        
        self.attention = nn.MultiheadAttention(embed_dim=channels, 
                                               bias=True, 
                                               batch_first=True,
                                               num_heads=2)

    def forward(self, x):
        att_value = self.att_value(x)
        att_value = att_value.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
        att_value = att_value.permute(0, 2, 1)
        x1 = self.attention(query=att_value, value=att_value, key=att_value, need_weights=False)
        
        x1 = x1[0].permute(0, 2, 1)
        x1 = x1.view(x1.shape[0], x1.shape[1], np.sqrt(x1.shape[2]).astype(int), np.sqrt(x1.shape[2]).astype(int))

        return x1
    
class Dilate_fusion(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, 1, padding="same")
        self.conv_dil2 = nn.Conv2d(in_channel, out_channel, 3, 1, padding="same", dilation=2)
        self.conv_dil3 = nn.Conv2d(in_channel, out_channel, 3, 1, padding="same", dilation=3)

        self.Ge_Drp = nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self,x):
        x1 = self.conv(x)
        x1 = self.Ge_Drp(x1)

        x2 = self.conv_dil2(x)
        x2 = self.Ge_Drp(x2)

        x3 = self.conv_dil3(x)
        x3 = self.Ge_Drp(x3)

        add = torch.add(x1,x2)
        add = torch.add(add,x3)

        x_out = self.conv(add)
        x_out = self.Ge_Drp(x_out)

        return x_out

class EAF(nn.Module):
    """
    高效特征融合模块，专门为多尺度特征设计
    """
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        
        self.compressed_channels = in_channels // reduction
        # 通道压缩
        self.compression = nn.Sequential(
            nn.Conv2d(in_channels, self.compressed_channels, 1),
            nn.BatchNorm2d(self.compressed_channels),
            nn.ReLU(inplace=True)
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.compressed_channels, self.compressed_channels, 1),
            nn.Sigmoid()
        )
        
        # 扩展回原通道数
        self.expansion = nn.Sequential(
            nn.Conv2d(self.compressed_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 压缩通道
        compressed = self.compression(x)
        
        # 空间注意力
        avg_out = torch.mean(compressed, dim=1, keepdim=True)
        max_out, _ = torch.max(compressed, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_att)
        
        # 通道注意力
        channel_att = self.channel_attention(compressed)
        
        # 应用注意力
        attended = compressed * spatial_att * channel_att
        
        # 扩展回原通道数
        expanded = self.expansion(attended)
        
        return expanded + x  # 残差连接

class CADF(nn.Module):
    """
    CADF
    input:x.shape = [b, in_channel, h, w]
    output:x.shape = [b, out_channel, h\2, w\2]
    """
    def __init__(self, 
                 in_channel,
                 out_channel):
        super().__init__()

        # self.attention_out = Attention(out_channel)
        self.attention_out = XFormersAttention(out_channel)
        self.layernorm_in = Layernorm(in_channel, eps=1e-5)
        self.layernorm_out = Layernorm(out_channel, eps=1e-5)
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, padding="same")
        self.dilate_fusion = Dilate_fusion(out_channel,out_channel)
        self.downsampling = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d((2,2))
        )

    def forward(self,x):
        x = self.layernorm_in(x)
        x = self.downsampling(x)

        x1 = self.attention_out(x)
        x1 = self.conv2(x1)

        x2 = torch.add(x1,x)

        x3 = self.layernorm_out(x2)
        x3 = self.dilate_fusion(x3)

        x3 = torch.add(x2,x3)

        return x2

class SSCADFNet(nn.Module):
    def __init__(self):
        super().__init__()

        feas = [8,16,32,64,128]

        encoder = []
        conv1s = []
        dysamples = []
        in_channel = 3
        for i,fea in enumerate(feas):
            encoder.append(CADF(in_channel,fea))
            in_channel = fea

            conv1s.append(nn.Conv2d(fea,8,1))
            
            scale = 2**(i+1)
            dysamples.append(DySample(8,scale=scale))

        self.encoder = nn.ModuleList(encoder)
        self.conv1s = nn.ModuleList(conv1s)
        self.dysamples = nn.ModuleList(dysamples)
        self.EAF = EAF(8*5)
        self.dropout = nn.Dropout(p=0.1)

        self.linear_fuse = BottConv(8*5, 8, 1, kernel_size=1, padding=0, stride=1)

        self.linear_pred = BottConv(8, 2, 1, kernel_size=1)


    def forward(self,x):
        
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
        
        f1,f2,f3,f4,f5 = skips # [1, 8, 256, 256],[1, 16, 128, 128],[1, 32, 64, 64],[1, 64, 32, 32],[1, 128, 16, 16]

        outs = []
        for skip,conv1,dysample in zip(skips,self.conv1s,self.dysamples):
            out = conv1(skip)
            out = dysample(out)
            outs.append(out)
        
        out = torch.cat(outs,dim=1)
        out = self.EAF(out)

        out = self.linear_fuse(out)

        out = self.dropout(out)
        x = self.linear_pred(out)

        return x


from thop import profile, clever_format

def calculate_params_flops(model, input_shape=(1, 3, 224, 224), device='cuda'):
    """
    计算模型的参数量和FLOPs
    
    Args:
        model: PyTorch模型
        input_shape: 输入张量形状 (batch_size, channels, height, width)
        device: 计算设备
    """
    # 创建随机输入
    dummy_input = torch.randn(input_shape).to(device)
    model = model.to(device)
    model.eval()
    
    # 计算FLOPs和参数量
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    # 格式化输出
    flops, params = clever_format([flops, params], "%.3f")
    
    print(f"Parameters: {params}")
    print(f"FLOPs: {flops}")
    
    return flops, params

import time
def measure_inference_speed(model, input_shape=(1, 3, 512, 512), num_runs=10, warmup=3, device='cuda'):
    """
    基础推理速度测量
    """
    model = model.to(device)
    model.eval()
    
    # 创建输入张量
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warm-up
    print("Warming up...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # 测量推理时间
    print("Measuring inference speed...")
    latencies = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize()  # 等待CUDA操作完成
            end_time = time.time()
            
            latencies.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    # 计算统计信息
    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    
    # 计算FPS
    fps = 1000 / mean_latency
    
    print(f"\n=== 推理速度分析 ===")
    print(f"测试次数: {num_runs}")
    print(f"输入形状: {input_shape}")
    print(f"平均延迟: {mean_latency:.2f} ms")
    print(f"延迟标准差: {std_latency:.2f} ms")
    print(f"最小延迟: {min_latency:.2f} ms")
    print(f"最大延迟: {max_latency:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"99%延迟: {np.percentile(latencies, 99):.2f} ms")
    
    return mean_latency, fps

if __name__ == '__main__':

    device = torch.device('cuda')
    model = SSCADFNet().to(device)
    # x = torch.randn(1, 3, 512, 512).to(device)
    # with torch.no_grad():
    #     y = model(x)
    # y = y.argmax(1)
    # print(y.shape)
    
    calculate_params_flops(model, input_shape=(1, 3, 512, 512))
    # measure_inference_speed(model, input_shape=(1, 3, 512, 512))
    
    # feas = [8,16,32,64,128]
    # in_channel = 3
    
    # skips = []
    # for fea in feas:
    #     ct = CTBlock(in_channel,fea).to(device)
    #     # mlp = MLP(input_dim=fea, embed_dim=8)
    #     x = ct(x)
    #     skips.append(x)
    #     # print(x.shape)
    #     in_channel = fea
        
    # out1s = []
    # for fea,skip in zip(feas,skips):
    #     mlp = MLP(input_dim=fea, embed_dim=8).to(device)
    #     b,c,h,w = skip.shape
    #     out = mlp(skip.reshape(b, c, h*w).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, 8, h, w)
    #     out1s.append(out)
    #     # print(out.shape)
    
    # out2s = []
    # for i,out in enumerate(out1s):
    #     scale = 2**(i+1)
    #     dysample = DySample(8,scale=scale).to(device)
    #     out = dysample(out)
    #     out2s.append(out)   
    #     # print(out.shape) 
    
    # gbc = GBC(8*5).to(device)
    # out = gbc(torch.cat(out2s,dim=1))
    
    # linear_fuse = BottConv(8*5, 8, 1, kernel_size=1, padding=0, stride=1).to(device)
    # dropout = nn.Dropout(p=0.1).to(device)
    # linear_pred = BottConv(8, 2, 1, kernel_size=1).to(device)
    # linear_pred_1 = nn.Conv2d(2, 2, kernel_size=1).to(device)
    
    
    # out = linear_fuse(out)
    # # print(out.shape)
    # out = dropout(out)
    # # print(out.shape)
    # x = linear_pred_1(linear_pred(out))
    # print(x)