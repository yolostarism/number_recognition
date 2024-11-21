import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


# 定义模型类（与训练时的模型结构一致）
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 9216)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# 加载模型参数
model = CNN()
model.load_state_dict(torch.load('mnist_cnn.pth', weights_only=True))
model.eval()

# 定义图像预处理转换
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 如果是彩色图像，转换为灰度图像
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def preprocess_image(image_path):
    # 打开图像
    image = Image.open(image_path)
    # 应用转换
    image = transform(image)
    # 增加批次维度
    image = image.unsqueeze(0)
    return image


# 示例：预处理并预测一张图像
image_path = 'handwritten_7.png'
image_tensor = preprocess_image(image_path)

with torch.no_grad():
    output = model(image_tensor)
    predicted_label = output.argmax(dim=1, keepdim=True).item()

print(f'预测标签: {predicted_label}')
