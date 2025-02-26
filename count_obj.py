def count_lines_per_object(obj_file_path):
    object_lines = {}
    current_object = None
    line_count = 0

    try:
        with open(obj_file_path, 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if line.startswith('o '):
                    if current_object is not None:
                        object_lines[current_object] = line_count
                    current_object = line.split(' ', 1)[1].strip()
                    line_count = 0
                line_count += 1

            # 处理最后一个物体
            if current_object is not None:
                object_lines[current_object] = line_count

    except FileNotFoundError:
        print(f"文件 {obj_file_path} 未找到。")
        return

    return object_lines

# 示例使用
obj_file_path = '/ssd/yangyuqiang/infinigen/outputs/multi_dataset_no_plantandshlefobj/506fc8f/fine/scene.obj'
result = count_lines_per_object(obj_file_path)

# 计算总行数
total_lines = sum(result.values())

# 按照行数降序排序
sorted_result = sorted(result.items(), key=lambda item: item[1], reverse=True)
total_pro = 0.0
for object_name, line_count in sorted_result:
    # 计算单个物体占总物体大小的比例
    proportion = (line_count / total_lines) * 100
    total_pro += proportion
    print(f"物体 {object_name} 所占行数: {line_count}，占比: {proportion:.5f}%, cumsum: {total_pro:.5f}% ")