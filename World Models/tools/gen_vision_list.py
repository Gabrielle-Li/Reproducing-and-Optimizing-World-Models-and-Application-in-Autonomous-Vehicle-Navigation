import os

def write_image_paths(directory, output_file):

    with open(output_file, 'w') as f:

        for root, dirs, files in os.walk(directory):
            if files:  # 只处理非空文件夹
                for file in files:
                    if file.endswith('.jpg') or file.endswith('.txt'):  # 只处理jpg和png格式的图片
                       
                        image_path = os.path.join(root, file)
                        image_path = os.path.abspath(image_path)  # 将相对路径转换为绝对路径
                        f.write(f"{image_path}\n")     
                        


# # 指定目录和输出文件路径
# directory = 'train/map'  # 实际的目录路径
# output_file = 'list/map_list.txt'  # 实际的输出文件路径

directory = 'train/map_val'  # 实际的目录路径
output_file = 'list/map_val_list.txt'  # 实际的输出文件路径

# directory = 'train/vision'  # 实际的目录路径
# output_file = 'list/vision_list.txt'  # 实际的输出文件路径

# directory = 'train/vision_val'  # 实际的目录路径
# output_file = 'list/vision_val_list.txt'  # 实际的输出文件路径


# 调用函数写入图片路径
write_image_paths(directory, output_file)
