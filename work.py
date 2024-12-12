#
# import numpy as np
#
# def is_convergent(matrix, front_matrix):
#     m, n = matrix.shape
#     conv1 = True
#     conv2 = False
#     for i in range(m):
#         for j in range(i):
#             if(matrix[i][j] > 0.01):
#                 conv1 = False
#                 break
#         if conv1 == False:
#             break
#     conv2 = np.all(np.abs(matrix - front_matrix) < 0.01)
#     return conv1 or conv2
#
# def QR_decomposition(matrix):
#     m, n = matrix.shape
#     # 先求Q矩阵
#     while(is_convergent() != True)
#     for i in range(m):
#         a_i = matrix[i]
#         a = matrix[i]
#         for j in range(i):
#             a -= np.dot(a_i, matrix[j])*matrix[j]
#         matrix[i] = a/np.linalg.norm(a)
#

import json

# 定义文件路径
input_json_path = "D:\\迅雷下载\\translation2019zh\\translation2019zh_train.json"  # 替换为您的.json文件路径
output_txt_path = "D:\\CODE_REPOSITORY\\AI引论\\作业\\2023200440_李应_lab7\\transformer-nmt-pub\\nmt\\en-cn\\train1.txt"  # 替换为输出.txt文件的路径
num_records = 50000  # 需要读取的记录数


# 开始转换过程
def convert_json_to_txt_with_tab(input_path, output_path, num_records):
    with open(input_path, 'r', encoding='utf-8') as json_file, \
            open(output_txt_path, 'w', encoding='utf-8') as txt_file:

        for i, line in enumerate(json_file):
            if i >= num_records:
                break  # 如果达到指定的记录数，则停止读取

            data = json.loads(line)
            english = data.get('english', '')
            chinese = data.get('chinese', '')

            # 使用tab作为分隔符
            txt_file.write(f"{english}\t{chinese}\n")


# 运行转换函数
convert_json_to_txt_with_tab(input_json_path, output_txt_path, num_records)

