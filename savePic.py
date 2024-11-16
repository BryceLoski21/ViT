from torchvision import datasets, transforms
from PIL import Image
import os


# 创建保存图片的目录
save_dir = './data/mnist_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 定义转换操作，将图片转换为Tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# 加载测试集
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# 选择要保存的图片数量
num_images = 10

# 遍历前10张图片并保存
for i, (image, label) in enumerate(test_dataset):
    if i >= num_images:
        break
    # 将Tensor转换为PIL图像
    image = image.squeeze()  # 去掉批次维度
    image = Image.fromarray(image.numpy().astype('uint8'), 'L')  # 转换为灰度图

    # 保存图片
    image.save(os.path.join(save_dir, f'image_{i+1}_label_{label}.png'))