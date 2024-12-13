import json

def convert_list_to_lines(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    
    with open(output_file, 'w', encoding='utf-8') as fout:
        for word in data:
            fout.write(f"{word}\n")
    
    print(f"Converted {input_file} to {output_file} with {len(data)} words.")

if __name__ == '__main__':
    # 转换 wordList.txt
    convert_list_to_lines('../wordList.txt', '../wordList_converted.txt')
    
    # 转换 idiomList.txt
    convert_list_to_lines('../idiomList.txt', '../idiomList_converted.txt')