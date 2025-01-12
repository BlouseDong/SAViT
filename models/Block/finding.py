import json

def find_files_with_points(json_file, target_points_count):
    # 打开 JSON 文件
    with open(json_file, 'r') as file:
        data = json.load(file)

    # 遍历数据，查找 'points' 个数为 target_points_count 的文件
    matching_files = []
    for file_name, details in data.items():
        if isinstance(details, dict) and 'points' in details:
            points_count = len(details['points'])
            if points_count == target_points_count:
                matching_files.append(file_name)

    return matching_files

# 文件路径和目标 points 数量
json_file = r'D:\zhiwei\CACViT\Dataset\FSC147\annotation_FSC147_384.json'
target_points_count =124

# 查找符合条件的文件名
result_files = find_files_with_points(json_file, target_points_count)

# 输出结果
if result_files:
    print("Files with 'points' count equal to", target_points_count, ":")
    for file in result_files:
        print(file)
else:
    print("No files with 'points' count equal to", target_points_count, "found.")
