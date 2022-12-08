from datasets import load_dataset
from transformers import BertTokenizer

def load_train_data(filename):
    data = load_dataset(filename, split='train[:100]')
    return data

def tokenize_data(data):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens = data.map(lambda x: {'text': tokenizer.encode(x['text'])})
    return tokens
