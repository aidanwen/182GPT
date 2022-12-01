import pandas
import spacy

def load_data(filename):
    data=pandas.read_csv(filename,sep='\t')
    return data

def tokenize_data(data):
    nlp = spacy.load("en_core_web_sm")
    tokens = [ent.text for ent in doc.ents for doc in nlp.pipe(data)]
    return tokens
