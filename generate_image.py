from PIL import Image, ImageDraw, ImageFont

# 创建一个28x28的空白图像，背景为白色
image = Image.new('L', (28, 28), color=255)
draw = ImageDraw.Draw(image)

# 使用Pillow的默认字体
font = ImageFont.load_default()

# 在图像中央绘制一个“7”
draw.text((8, 4), '7', fill=0, font=font)

# 保存图像
image.save('handwritten_7.png')

# 显示图像
image.show()
