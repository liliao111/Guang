import warnings
warnings.filterwarnings('ignore', message='.*is not compatible with the current PyTorch installation.*')
import torch
import time

# CPU 测试
print('CPU Test:')
start = time.time()
x_cpu = torch.randn(10000, 10000)
y_cpu = torch.randn(10000, 10000)
z_cpu = torch.matmul(x_cpu, y_cpu)
cpu_time = time.time() - start
print(f'CPU Time: {cpu_time:.3f}s')
        
# GPU 测试
print('\nGPU Test:')
start = time.time()
x_gpu = torch.randn(10000, 10000).cuda()
y_gpu = torch.randn(10000, 10000).cuda()
z_gpu = torch.matmul(x_gpu, y_gpu)
torch.cuda.synchronize()
gpu_time = time.time() - start
print(f'GPU Time: {gpu_time:.3f}s')

print(f'\nGPU Speedup: {cpu_time/gpu_time:.1f}x')