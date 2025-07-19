import re
# import matplotlib.pyplot as plt

# import torch
# import spacy
# from razdel import sentenize

# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from scipy.signal import savgol_filter


with open('text.txt', encoding='utf-8') as file:
    text = file.read()


def clean_text(text: str) -> str:
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()

    return text


cleaned_text = clean_text(text)

# [1] Using spacy

# nlp = spacy.load('en_core_web_sm')
# doc = nlp(cleaned_text)

# sentences = [sentence for sentence in doc.sents]


# [2] Using razdel

# sentences = [sentence.text for sentence in list(sentenize(cleaned_text))]

# model_checkpoint = 'cointegrated/rubert-tiny-sentiment-balanced'
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)