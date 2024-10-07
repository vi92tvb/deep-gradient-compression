import os
from ..dataset import Dataset
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import regex as re
from torch.nn.utils.rnn import pad_sequence
import underthesea
import numpy as np
from torch.utils.data import TensorDataset
import torch
import torchtext.vocab as vocab
from underthesea import word_tokenize


__all__ = ['VietnameseText']

class VietnameseText(Dataset):
    def __init__(self, root):
        self.root = root
        self.vocab = Vocabulary()
        self.pad_idx = self.vocab["<pad>"]

        dataset_dict = {'train': [], 'test': [], 'val': []}

        folder_order = ['train', 'test', 'val']

        word_embedding = vocab.Vectors(name = "vi_word2vec.txt",
                               unk_init = torch.Tensor.normal_)

        words_list = list(word_embedding.stoi.keys())

        # Limit word, not enough space
        for word in words_list[:1587507]:
            self.vocab.add(word)

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

                data['sentiment'] = data['label'].progress_apply(transform_label)
                data['processed'] = self.vocab.tokenize_corpus(data['text'])
                labels = torch.tensor(data['label'].to_numpy(), dtype=torch.long)
            
                tensor_data = self.vocab.corpus_to_tensor(data['processed'], is_tokenized=True)

                padded_corpus = pad_sequence(tensor_data, batch_first=True, padding_value=self.pad_idx)

                result = TensorDataset(padded_corpus, labels)

                dataset_dict[subdir] = result

        super().__init__(train=dataset_dict['train'], val=dataset_dict['val'], test=dataset_dict['test'])
        self.dataset_dict = {'train': dataset_dict['train'], 'test': dataset_dict['test'], 'val': dataset_dict['val']} 

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


class Vocabulary:
    """ The Vocabulary class is used to record words, which are used to convert
        text to numbers and vice versa.
    """

    def __init__(self):
        self.word2id = dict()
        self.word2id['<pad>'] = 0   # Pad Token
        self.word2id['<unk>'] = 1   # Unknown Token
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __len__(self):
        return len(self.word2id)

    def id2word(self, word_index):
        """
        @param word_index (int)
        @return word (str)
        """
        return self.id2word[word_index]

    def add(self, word):
        """ Add word to vocabulary
        @param word (str)
        @return index (str): index of the word just added
        """
        if word not in self:
            word_index = self.word2id[word] = len(self.word2id)
            self.id2word[word_index] = word
            return word_index
        else:
            return self[word]

    @staticmethod
    def tokenize_corpus(corpus):
        """Split the documents of the corpus into words
        @param corpus (list(str)): list of documents
        @return tokenized_corpus (list(list(str))): list of words
        """
        print("Tokenize the corpus...")
        if isinstance(corpus, np.ndarray):
            corpus = corpus.tolist()
        tokenized_corpus = list()
        for document in tqdm(corpus):
            tokenized_document = [word.replace(" ", "_") for word in word_tokenize(document)]
            tokenized_corpus.append(tokenized_document)

        return tokenized_corpus

    def corpus_to_tensor(self, corpus, is_tokenized=False):
        """ Convert corpus to a list of indices tensor
        @param corpus (list(str) if is_tokenized==False else list(list(str)))
        @param is_tokenized (bool)
        @return indicies_corpus (list(tensor))
        """
        if is_tokenized:
            tokenized_corpus = corpus
        else:
            tokenized_corpus = self.tokenize_corpus(corpus)
        indicies_corpus = list()
        for document in tqdm(tokenized_corpus):
            indicies_document = torch.tensor(list(map(lambda word: self[word], document)),
                                             dtype=torch.int64)
            indicies_corpus.append(indicies_document)

        return indicies_corpus

    def tensor_to_corpus(self, tensor):
        """ Convert list of indices tensor to a list of tokenized documents
        @param indicies_corpus (list(tensor))
        @return corpus (list(list(str)))
        """
        corpus = list()
        for indicies in tqdm(tensor):
            document = list(map(lambda index: self.id2word[index.item()], indicies))
            corpus.append(document)

        return corpus

def get_vector(embeddings, word):
    """ Get embedding vector of the word
    @param embeddings (torchtext.vocab.vectors.Vectors)
    @param word (str)
    @return vector (torch.Tensor)
    """
    assert word in embeddings.stoi, f'*{word}* is not in the vocab!'
    return embeddings.vectors[embeddings.stoi[word]]

def closest_words(embeddings, vector, n=10):
    """ Return n words closest in meaning to the word
    @param embeddings (torchtext.vocab.vectors.Vectors)
    @param vector (torch.Tensor)
    @param n (int)
    @return words (list(tuple(str, float)))
    """
    distances = [(word, torch.dist(vector, get_vector(embeddings, word)).item())
                 for word in embeddings.itos]

    return sorted(distances, key = lambda w: w[1])[:n]

