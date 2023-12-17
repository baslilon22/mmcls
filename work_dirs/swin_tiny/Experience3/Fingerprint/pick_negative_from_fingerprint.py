import json
import os
import shutil
import random
# 指定JSON文件路径
json_file = '/data4/lj/Classification/1.7/work_dirs/swin_tiny/Experience3/Fingerprint/xwmap.json'
destination_folder = "/data4/lj/Dataset/JML/Experience3/Fingerprint/negative"
# 打开JSON文件并解析数据
with open(json_file, 'r') as f:
    data = json.load(f)

for key, value in data.copy().items():
    if value[:2] != "yl":
        del data[key]
# shutil.move
Dataset_path = "/data4/lj/Dataset/drink_shrank/drink"
for key, value in data.items():
    Folder_name = "xw_label_" + key
    Folder_path = os.path.join(Dataset_path,Folder_name)
    # 获取文件夹下的所有文件名
    if os.path.exists(Folder_path):
        file_names = os.listdir(Folder_path)
        # 随机选择16张图片
        if len(file_names) > 19:
            selected_files = random.sample(file_names, 19)
            for i in selected_files:
                img_path = os.path.join(Folder_path,i)
                destination_path = os.path.join(destination_folder,i)
                shutil.copy(img_path,destination_path)
        else:
            for i in file_names:
                img_path = os.path.join(Folder_path,i)
                destination_path = os.path.join(destination_folder,i)
                shutil.copy(img_path,destination_path)
        
# 打印读取的JSON数据
# print(data)