import torch
import torchvision
import PIL
from PIL.Image import Image
from torch import nn
import time
import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from einops import rearrange
from torchvision import datasets, transforms
from model import ViT


model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,  # 模型
            dim=64, depth=6, heads=8, mlp_dim=128)

model.load_state_dict(torch.load('./model/model'))
model.eval()  # 设置为评估模式

# 定义预处理步骤
preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),  # 数据增强，ToTensor+Normalize
                                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))])

# 加载图片
img = PIL.Image.open("./data/mnist_images/image_1_label_5.png")

# 预处理图片
img_t = preprocess(img)

# 添加批次维度
batch_t = torch.unsqueeze(img_t, 0)

# 如果模型在GPU上训练，则将输入数据发送到GPU
if torch.cuda.is_available():
    batch_t = batch_t.to('cuda')
    model.to('cuda')

# 推理
with torch.no_grad():  # 在推理时不需要计算梯度
    output = model(batch_t)

# 定义一个从索引到标签的映射
class_to_label = {
    0: 'Zero',
    1: 'One',
    2: 'Two',
    3: 'Three',
    4: 'Four',
    5: 'Five',
    6: 'Six',
    7: 'Seven',
    8: 'Eight',
    9: 'Nine'
}

# 将输出转换为概率分布
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# 获取最可能的类别
_, predicted_class = torch.max(probabilities, 0)

# 将类别索引转换为类别标签
predicted_label = class_to_label[int(predicted_class)]

print(predicted_label)
