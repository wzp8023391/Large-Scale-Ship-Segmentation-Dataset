import torch
import numpy as np
from thop import profile, clever_format

from module.SSegCADFNet import SSCADFNet

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
   
    calculate_params_flops(model, input_shape=(1, 3, 512, 512))