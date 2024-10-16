import os

def write_image_paths(directory, output_file):
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(directory):
            if files:  # 只处理非空文件夹
                for file in sorted(files):  # 确保按照文件名的顺序处理
                    if file.endswith('.jpg') or file.endswith('.png'):  # 只处理jpg和png格式的图片
                        image_path = os.path.join(root, file)
                        image_path = os.path.abspath(image_path)  # 将相对路径转换为绝对路径

                        # 获取当前图片的数字编号
                        try:
                            current_image_number = int(os.path.splitext(file)[0].split("_")[-1])  # 假设文件名为数字.jpg或数字.png
                        except ValueError:
                            continue  # 跳过处理

                        # 添加限制，跳过大于等于999的图片名
                        if current_image_number >= 315:
                            continue

                        # 在第一次迭代时，previous_image_number为None，跳过此步骤
                        
                        future_image_number = current_image_number + 1
                        future_image_path = os.path.join(root, f"view_{future_image_number}.jpg")
                        future_image_path = os.path.abspath(future_image_path)
                        f.write(f"{image_path} {future_image_path}\n")
                        
                        
                        

                    

# # 指定目录和输出文件路径
# directory = 'train/screen'  # 实际的目录路径
# output_file = 'list/screen_list.txt'  # 实际的输出文件路径

# directory = 'train/screen_val'  # 实际的目录路径
# output_file = 'list/screen_val_list.txt'  # 实际的输出文件路径

# directory = 'train/vision'  # 实际的目录路径
# output_file = 'list/vision_memory.txt'  # 实际的输出文件路径

directory = 'train/vision_val'  # 实际的目录路径
output_file = 'list/vision_memory_val.txt'  # 实际的输出文件路径

# 调用函数写入图片路径
write_image_paths(directory, output_file)
