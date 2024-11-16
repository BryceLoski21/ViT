import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# 定义转换操作，将图片转换为Tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载测试集
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# 选择要显示的图片数量
num_images = 10

# 创建一个图形和子图
fig, axes = plt.subplots(1, num_images, figsize=(10, 2))

# 遍历每张图片并显示
for i, (image, label) in enumerate(test_dataset):
    ax = axes[i]
    ax.imshow(image.squeeze(), cmap='gray')  # 将Tensor转换为图片并显示
    ax.set_title(f'Label: {label}')
    ax.axis('off')  # 不显示坐标轴

    if i == num_images - 1:
        plt.show()
        break