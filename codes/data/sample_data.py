import random

def sample_file(input_file, output_file, sample_ratio=0.1):
    with open(input_file, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()

    total = len(lines)
    sample_size = int(total * sample_ratio)
    sampled_lines = random.sample(lines, sample_size)

    with open(output_file, 'w', encoding='utf-8') as fout:
        fout.writelines(sampled_lines)

    print(f'Sampled {sample_size} out of {total} lines from {input_file} to {output_file}')

if __name__ == '__main__':
    import os

    # 设置采样比例，例如10%
    sample_ratio_1 = 0.002
    sample_ratio_2 = 0.005

    # 确保输出目录存在
    output_dir = './'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 采样训练数据
    sample_file('./train_data.txt', './train_data_sampled.txt', sample_ratio_1)

    # 采样验证数据
    sample_file('./dev_data.txt', './dev_data_sampled.txt', sample_ratio_2)

    sample_file('./test_data.txt', './test_data_sampled.txt', sample_ratio_2)