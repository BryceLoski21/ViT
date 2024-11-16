import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm  # 用于加载 ViT 模型
from demo import ViT

# 数据预处理：将图像转换为 tensor 并进行归一化
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 调整为3通道
    transforms.Resize((224, 224)),  # 调整图像大小以适配 ViT 模型
    transforms.ToTensor(),          # 转换为 tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 加载训练和测试数据
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

