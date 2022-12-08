!pip install datasets
!pip install transformers


data = load_dataset("oscar", "unshuffled_deduplicated_azb")
tokens = tokenize_data(data)

def build_batch(tokens, batch_size):
    flat_tokens = list(np.array(tokens).flatten())
    indices = list(np.random.randint(0, len(tokens), size=batch_size,dtype=int))
    # print(indices)
    batch = [tokens[i] for i in indices]

    return batch

dataset = []
for seq in tokens['train']:
    dataset.extend(seq)
batch = build_batch(dataset, 512)
