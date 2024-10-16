from PIL import Image


def resize_image_keep_ratio(input_image_path, output_image_path, max_size):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    print(f"The original image size is {width} wide x {height} high")

    # 计算缩放比例，保持宽高比例不变
    ratio = min(max_size[0] / width, max_size[1] / height)
    new_size = (int(width * ratio), int(height * ratio))

    resized_image = original_image.resize(new_size)
    width, height = resized_image.size
    print(f"The resized image size is {width} wide x {height} high")

    resized_image.save(output_image_path)


if __name__ == '__main__':
    input_image = 'resource/car3.png'  # 指定输入图片路径
    output_image = 'resource/Car_3.png'  # 指定输出图片路径
    max_size = (100, 20)  # 指定最大的缩放尺寸，格式为 (最大宽度, 最大高度)

    resize_image_keep_ratio(input_image, output_image, max_size)
