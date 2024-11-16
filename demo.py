import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm  # 用于加载 ViT 模型


# 定义一个ViT 模型
class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器
        encoderLayer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=2048, dropout=0.1,
                                   activation='relu', layer_norm_eps=1e-5, batch_first=True,
                                   norm_first=False, bias=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoderLayer, num_layers=1, enable_nested_tensor=True, mask_check=True)
        # 解码器
        decoderLayer = nn.TransformerDecoderLayer(d_model=32, nhead=4, dim_feedforward=2048, dropout=0.1,
                                                   activation='relu', layer_norm_eps=1e-5, batch_first=True,
                                                   norm_first=False, bias=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoderLayer, num_layers=1)
        self.fc_out = nn.Linear(in_features=32, out_features=10)

    def forward(self, src, target):
        memory = self.encoder(src)
        output = self.decoder(target, memory)
        output = self.fc_out(output)
        return output



# test
src = torch.rand(32, 32, 32)
target = torch.rand(32, 32, 32)

model = ViT()
output = model(src, target)
print(output)

# model = ViT()
#
# # 替换最后一层，使其输出为 10 类（手写数字识别的类别数）
# model.head = nn.Linear(model.head.in_features, 10)
#
# # 使用交叉熵损失函数
# criterion = nn.CrossEntropyLoss()
#
# # 使用 Adam 优化器
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#

