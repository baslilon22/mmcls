import pandas as pd


# 打开txt文档
file_path = '/data4/lj/Classification/1.7/work_dirs/swin_tiny/Experience3/Test_Cluster/negative_example.txt'
file = open(file_path, 'r')

cluster_file_path = '/data4/lj/Classification/1.7/work_dirs/swin_tiny/Experience3/Test_Cluster/Super_model_positive_labels_k45.txt'
cluster_file = open(cluster_file_path, 'r')
# 读取文档内容并计算行数
class_lines = file.readlines()
# 去除行末的换行符
class_lines = [class_line.rstrip() for class_line in class_lines]
cluster_lines = cluster_file.readlines()
cluster_lines = [cluster_line.rstrip() for cluster_line in cluster_lines]

dict = {}
for i,class_line in enumerate(class_lines):
    class_name = class_line.split('/')[0]

    if class_name not in dict:
        dict[class_name] = {}
        dict[class_name][cluster_lines[i]] = 1 #初始化
    else:
        if cluster_lines[i] not in dict[class_name]:
            dict[class_name][cluster_lines[i]] = 1
        else:
            dict[class_name][cluster_lines[i]] += 1
# print(dict)
# 关闭文件
# 创建一个DataFrame对象
max = 0

for i in dict:
    for j in dict[i]:
       if dict[i][j] > max:
            max_index = j
            max = dict[i][j]
    dict[i] = max_index
    max = 0
df = pd.DataFrame.from_dict(dict, orient='index')

# 将每个子字典按值进行降序排序
df = df.apply(lambda row: sorted(row.items(), key=lambda x: x[1], reverse=True), axis=1)

# 保存为xls文档
df.to_excel('work_dirs/swin_tiny/Experience3/Test_Cluster/positive_output.xlsx', index_label='Key')
file.close()

# 打印行数
# print('文档行数:', num_lines)