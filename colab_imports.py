!pip install datasets
!pip install transformers


data = load_dataset("oscar", "unshuffled_deduplicated_azb")
tokens = tokenize_data(data)
