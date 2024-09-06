import os
from ..dataset import Dataset
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import regex as re
import string
import underthesea
from collections import Counter
import numpy as np
from torch.utils.data import TensorDataset
import torch

__all__ = ['VietnameseText']

class VietnameseText(Dataset):
    def __init__(self, root):
        self.root = root
        dataset_dict = {'train': [], 'test': [], 'val': []}

        folder_order = ['train', 'test', 'val']
        vocab = {'<PAD>': 0, '<UNK>': 1}
        
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
                sentences = [sentence.strip() for sentence in sentences]

                # sentiments = [int(sentiment.strip()) for sentiment in sentiments]

                sentiments = [int(sentiment.strip()) for sentiment in sentiments]
                data = pd.DataFrame({
                    'text': sentences,
                    'label': sentiments
                })
                # Drop rows where any element is NaN
                data = data.dropna()

                # Drop rows where 'text' or 'label' is empty
                data = data[(data['text'].str.strip() != '') & (data['label'].notna())]

                dataset_dict[subdir] = data

                if subdir == "train":
                    data['sentiment'] = data['label'].progress_apply(transform_label)
                    data['processed'] = data['text'].progress_apply(clean_document)
        
                    reviews = data['processed'].values
                    reviews = [' '.join(review) for review in reviews]  # Join tokens to form the processed sentences
                    words = ' '.join(reviews)
                    words = words.split()

                    # Create vocabulary and word to index mappings
                    all_words = ' '.join(reviews).split()
                    counter = Counter(all_words)
                    vocab = sorted(counter, key=counter.get, reverse=True)
                    int2word = {i + 2: word for i, word in enumerate(vocab)}
                    int2word[0] = '<PAD>'
                    int2word[1] = '<UNK>'
                    self.word2int = {word: id for id, word in int2word.items()}

        super().__init__(train=dataset_dict['train'], val=dataset_dict['val'], test=dataset_dict['test'])
        self.dataset_dict = {'train': dataset_dict['train'], 'test': dataset_dict['test'], 'val': dataset_dict['val']} 

    def __len__(self):
        return sum(len(self.dataset_dict[key]) for key in self.dataset_dict)

    def __getitem__(self, index):
        data = self.dataset_dict[index]
        data['sentiment'] = data['label'].progress_apply(transform_label)
        data['processed'] = data['text'].progress_apply(clean_document)
        reviews = data['processed'].values
        reviews = [' '.join(review) for review in reviews]

        reviews_enc = [[self.word2int.get(word, self.word2int['<UNK>']) for word in review.split()] for review in tqdm(reviews)]
        seq_length = 256
        features = pad_features(reviews_enc, pad_id=self.word2int['<PAD>'], seq_length=seq_length)
        labels = data['label'].to_numpy()
        dataset = TensorDataset(torch.tensor(features, dtype=torch.long), torch.tensor(labels, dtype=torch.long))
        return dataset


def build_vocab(data):
  """
  Builds vocabulary from the text data.
  """
  word_counts = Counter()
  for text in data['text']:
    word_counts.update(text.split())
  vocab = {'<PAD>': 0, '<UNK>': 1}
  vocab.update({word: i for i, (word, count) in enumerate(word_counts.most_common())})
  return vocab

def transform_label(label):
    if label == 0: return 'neg'
    if label == 1: return 'neu'
    if label == 2: return 'pos'

def pad_features(reviews, pad_id, seq_length=50):
    features = np.full((len(reviews), seq_length), pad_id, dtype=int)

    for i, row in enumerate(reviews):
        # if seq_length < len(row) then review will be trimmed
        features[i, :len(row)] = np.array(row)[:seq_length]

    return features

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

def clean_document(doc):
    doc.lower()
    tokens = underthesea.word_tokenize(doc) #Pyvi Vitokenizer library
    tokens = [token.lower().replace(" ", "_") for token in tokens]
    tokens = [word for word in tokens if word]
    return tokens
