import numpy as np

def load_vocab(vocab_file):
    vocab = {}
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            word = line.strip()
            vocab[word] = idx
    return vocab

def extract_embeddings(full_embed_file, vocab, output_file, embed_dim=300, skip_header=True):
    embed_matrix = np.random.uniform(-0.1, 0.1, (len(vocab), embed_dim)).astype(np.float32)
    found = 0

    with open(full_embed_file, 'r', encoding='utf-8') as f:
        if skip_header:
            next(f)  # 跳过头部行
        for line in f:
            parts = line.rstrip().split(' ')
            word = parts[0]
            if word in vocab:
                try:
                    vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    if len(vector) == embed_dim:
                        embed_matrix[vocab[word]] = vector
                        found += 1
                except ValueError:
                    print(f"Skipping line due to ValueError: {line[:50]}...")
                    continue

    print(f'Found {found}/{len(vocab)} words in the embedding.')

    np.save(output_file, embed_matrix)

if __name__ == '__main__':
    import os

    # 确保cache目录存在
    if not os.path.exists('cache'):
        os.makedirs('cache')

    # 处理词汇表
    word_vocab = load_vocab('./wordList_converted.txt')  # 使用转换后的词汇表
    extract_embeddings('./data/wordvector.txt', word_vocab, 'cache/word_embed_matrix.npy')

    # 处理成语表
    idiom_vocab = load_vocab('./idiomList_converted.txt')  # 使用转换后的成语表
    extract_embeddings('./data/wordvector.txt', idiom_vocab, 'cache/idiom_embed_matrix.npy')