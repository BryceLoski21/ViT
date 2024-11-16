# README

## 基于ViTransformer的手写数字识别

- 数据集Mnist
- 模型ViT
- 使用方法：
    1. 安装依赖`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
    2. 运行savePic.py：保存测试集前10张图片
    3. 运行eval.py: 对图片(5)进行推理

## 项目架构

```
Root[Root]------\
                |---data
                |     |---MNIST: 存放的Mnist数据集
                |     |
                |     |---mnist_images: 测试集中前10张图片
                |
                |---model: 保存的模型model
                |
                |---model.py: 模型的定义
                |
                |---savePic.py: 取测试集中的前10张图片
                |
                |---showPic。py: 显示测试集中的前10张图片
                |
                |---eval.py: 对示例图片进行预测
                |
                |---vit_mnist_print.txt: 保存训练的结果
                |
                |---requirements.txt: 依赖项
```

## ViT模型架构

```
ViT-----\
        |-----pos_embedding: 1 * (num_patch+1) * dim      
        |
        |-----patch_to_embedding: 将patch_dim（原图）经过embedding后得到dim维的嵌入向量
        |           |
        |           |-----Linear层(patch_dim, dim)
        |           
        |-----cls_token(1 * 1 * dim)           
        |   
        |-----Transformer(dim, depth, heads, mlp_dim): 叠加ModuleList，两个Res为一个注意力块
        |           |
        |           |-----Resdual_1 (PreNorm(dim, Attention))
        |           |       |
        |           |       |-----PreNorm层 (PreNorm(dim, Attention))
        |           |               |
        |           |               |-----LayerNorm层
        |           |               |-----Attention
        |           |
        |           |-----Resdual_2 (PreNorm(dim, FeedForward))
        |                   |
        |                   |-----PreNorm层 (PreNorm(dim, FeedForward))
        |                           |-----LayerNorm层
        |                           |-----FeedForward
        |                                   |-----Linear(dim, hidden_dim)
        |                                   |-----GELU
        |                                   |-----Linear(hidden_dim, dim)
        |
        |-----to_cls_token: Identiiy(), 恒等操作，直接返回输入张量
        |
        |-----MLP层
                |
                |-----Linear层(dim, mlp_dim)
                |-----激活函数GELU
                |-----Linear层(mlp_dim, num_classes)
```

## 参数

- `image_size`: int.  
    > Image size. If you have rectangular images, make sure your image size is the maximum of the width and height
- `patch_size`: int.  
    > Size of patches. `image_size` must be divisible by `patch_size`.  
    > The number of patches is: ` n = (image_size // patch_size) ** 2` and `n` **must be greater than 16**.
- `num_classes`: int.  
    > Number of classes to classify.
- `dim`: int.  
    > Last dimension of output tensor after linear transformation `nn.Linear(..., dim)`.
- `depth`: int.  
    > Number of Transformer blocks.
- `heads`: int.  
    > Number of heads in Multi-head Attention layer. 
- `mlp_dim`: int.  
    > Dimension of the MLP (FeedForward) layer. 
- `channels`: int, default `3`.  
    > Number of image's channels. 
- `dropout`: float between `[0, 1]`, default `0.`.  
    > Dropout rate. 
- `emb_dropout`: float between `[0, 1]`, default `0`.  
    > Embedding dropout rate.
- `pool`: string, either `cls` token pooling or `mean` pooling
