import torch
import torchvision
from torch import nn
import time
import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from einops import rearrange


# 定义参数
DOWNLOAD_PATH = 'data'
OUTPUT_PATH = 'vit_mnist_print.txt'
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 1000
N_EPOCHS = 5


# 打印结果以及输出到文件中
def print_info(string, file='vit_print.txt'):
    print(string)
    with open(file, 'a') as f:
        f.write(string + '\n')


# 残差模块，放在每个前馈网络和注意力之后
class Residual(nn.Module):  # 通过连接层或者补充，保证fn输出和x是同维度的
    def __init__(self, fn):  # 带function参数的Module，都是嵌套的Module
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# layernorm归一化,放在多头注意力层和激活函数层。用绝对位置编码的BERT，layernorm用来自身通道归一化
class PreNorm(nn.Module):  # 先归一化，再用function作用。
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 三维的用dim，四维用[C,H,W]
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# 放置多头注意力后，因为在于多头注意力使用的矩阵乘法为线性变换，后面跟上由全连 接网络构成的FeedForward增加非线性结构
class FeedForward(nn.Module):  # 非线性前馈，保持dim维不变
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


# 多头注意力层，多个自注意力连起来。使用qkv计算
class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])  # ModuleList套ModuleList
        for _ in range(depth):  # 叠加Attention块
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


# 将图像切割成一个个图像块,组成序列化的数据输入Transformer执行图像分类任务。
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, '报错：图像没有被patch_size完美分割'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2  # (P**2 C)：一个patch展平为向量后实际的长度

        self.patch_size = patch_size
        # 维度看起来比较奇怪，dim,num_patches+1就可以解决问题
        # 还要加个1维度，还要把别的反过来，是为了适应b（batch）维度，进行广播
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # +1是为了适应cls_token
        self.patch_to_embedding = nn.Linear(patch_dim, dim)  # 将patch_dim（原图）经过embedding后得到dim维的嵌入向量
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, mask=None):
        p = self.patch_size
        print('init', img.shape)

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)  # 将H W C 转化成 N (P P C)
        print('rearrange', x.shape)
        x = self.patch_to_embedding(x)  # 将(PPC)通过Embedding转化成一维embedding，这里的patch_to_embedding
        print('patch_embedding', x.shape)
        # 到这里，一张图片就和nlp里的一个句子以同样的形式输入Transformer中
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)  # batch长度不确定，cat没有广播机制，所以要expand
        print('cls_tokens', cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=1)  # 将类别信息接入embedding，0维是样本，1维是patch
        print('cat cls', x.shape)
        print('pos_embedding', self.pos_embedding.shape)
        x += self.pos_embedding  # +有广播机制，所以不需要expand
        x = self.transformer(x, mask)  # 送入encoder
        print('after transformer', x.shape)
        x = self.to_cls_token(x[:, 0])  # 取出class对应的token，用Identity占位
        print('cls_token', x.shape)
        y = self.mlp_head(x)  # 送入mlp分类器
        print('mlp_head', y.shape)
        return y


def train_epoch(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print_info('[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                       ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                       '{:6.4f}'.format(loss.item()), OUTPUT_PATH)
            loss_history.append(loss.item())


def evaluate(model, data_loader, loss_history):
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print_info('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
               '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
               '{:5}'.format(total_samples) + ' (' +
               '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n', OUTPUT_PATH)


torch.manual_seed(42)
# 加载数据
transform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),  # 数据增强，ToTensor+Normalize
                                                  # 组合操作：转换成张量。进行归一化操作，均值0.1307，标准差0.3081
                                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))])
# 训练集
train_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=True, download=True,
                                       transform=transform_mnist)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)  # 打乱数据
# 测试集
test_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=False, download=True,
                                      transform=transform_mnist)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)  # 打乱数据

'''
patch大小为 7x7（对于 28x28 图像，这意味着每个图像 4 x 4 = 16 个patch）、10 个可能的目标类别（0 到 9）和 1 个颜色通道（因为图像是灰度）。
在网络参数方面，使用了 64 个单元的维度，6 个 Transformer 块的深度，8 个 Transformer 头，MLP 使用 128 维度。
'''
model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,  # 模型
            dim=64, depth=6, heads=8, mlp_dim=128)
optimizer = optim.Adam(model.parameters(), lr=0.003)  # 优化器

start_time = time.time()  # 记录时间
train_loss_history, test_loss_history = [], []  # 精度记录
for epoch in range(1, N_EPOCHS + 1):
    temp_time = time.time()
    print_info(f'Epoch: {epoch}\n', OUTPUT_PATH)

    train_epoch(model, optimizer, train_loader, train_loss_history)  # 训练一epoch
    evaluate(model, test_loader, test_loss_history)  # 评估

    print_info(f'spend {(time.time() - temp_time) / 60} min\n', OUTPUT_PATH)

print_info('Execution time:' + '{:5.2f}'.format((time.time() - start_time) / 60) + ' min \n', OUTPUT_PATH)

# 指定保存文件的路径
model_path = './model/model'

# 保存整个模型
# 保存模型的state_dict
torch.save(model.state_dict(), model_path)
