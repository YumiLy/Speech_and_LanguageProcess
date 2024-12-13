import json

def validate_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error in line {i}: {line.strip()}")
                print(f"Error message: {e}")

validate_json('./train_data_sampled.txt')
validate_json('./dev_data_sampled.txt')
validate_json('./test_data_sample.txt')