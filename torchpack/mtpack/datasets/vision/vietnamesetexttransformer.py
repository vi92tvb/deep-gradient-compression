import os
from ..dataset import Dataset
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import regex as re
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.utils.data import TensorDataset
import torch
import torchtext.vocab as vocab
from underthesea import word_tokenize
from transformers import AutoModel, AutoTokenizer

__all__ = ['VietnameseTextTransformer']

class VietnameseTextTransformer(Dataset):
    def __init__(self, root):
        self.root = root
        dataset_dict = {'train': [], 'test': [], 'val': []}

        folder_order = ['train', 'test', 'val']

        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

        for subdir in folder_order:
            subdir_path = os.path.join(root, subdir)
            if os.path.isdir(subdir_path):
                sentences_path = os.path.join(subdir_path, 'sents.txt')
                sentiments_path = os.path.join(subdir_path, 'sentiments.txt')

                sentences = []
                sentiments = []

                with open(sentences_path, 'r', encoding='utf-8') as f:
                    sentences = f.readlines()
                with open(sentiments_path, 'r', encoding='utf-8') as f:
                    sentiments = f.readlines()

                sentences = [preprocess_text(sentence) for sentence in sentences]
                sentences = [standardize_data(sentence) for sentence in sentences] 
                sentences = [sentence.strip() for sentence in sentences]

                sentiments = [int(sentiment.strip()) for sentiment in sentiments]

                data = pd.DataFrame({
                    'text': sentences,
                    'label': sentiments
                })

                # Drop rows where any element is NaN
                data = data.dropna()

                # Drop rows where 'text' or 'label' is empty
                data = data[(data['text'].str.strip() != '') & (data['label'].notna())]

                data['sentiment'] = data['label'].progress_apply(transform_label)

                tqdm.pandas()
                data[['input_ids', 'attention_mask']] = data['text'].progress_apply(lambda x: pd.Series(tokenize_text(x)))

                labels = torch.tensor(data['label'].to_numpy(), dtype=torch.long)

                padded_input_ids = pad_sequence(data['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id)
                padded_marks = pad_sequence(data['attention_mask'], batch_first=True, padding_value=tokenizer.pad_token_id)

                result = TensorDataset(padded_input_ids, padded_marks, labels)
                dataset_dict[subdir] = result

        super().__init__(train=dataset_dict['train'], val=dataset_dict['val'], test=dataset_dict['test'])
        self.dataset_dict = {'train': dataset_dict['train'], 'test': dataset_dict['test'], 'val': dataset_dict['val']} 

def transform_label(label):
    if label == 0: return 'neg'
    if label == 1: return 'neu'
    if label == 2: return 'pos'

def tokenize_text(text):
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    line = word_tokenize(text, format="text")
    results = tokenizer.encode_plus(line, truncation=True, add_special_tokens=True, max_length=256, padding='max_length', return_attention_mask=True, return_token_type_ids=False)
    return torch.tensor(results['input_ids']), torch.tensor(results['attention_mask'])

def preprocess_text(text):
    # Define patterns to remove
    patterns_to_remove = [
        r'colonsmile', r'colonsad', r'colonsurprise', r'colonlove', r'colonsmilesmile', 
        r'coloncontemn', r'colonbigsmile', r'coloncc', r'colonsmallsmile', r'coloncolon',
        r'colonlovelove', r'colonhihi', r'doubledot', r'colonsadcolon', r'colonsadcolon', 
        r'colondoublesurprise', r'vdotv', r'dotdotdot', r'fraction', r'cshrap'
    ]
    
    # Remove each pattern from the text
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text)
    
    # Remove any extra spaces that may have been created
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def standardize_data(row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ")
    row = row.strip().lower()
    return row
