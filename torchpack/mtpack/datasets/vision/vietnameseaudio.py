import os
from sklearn.calibration import LabelEncoder
import torchaudio
import torch
from ..dataset import Dataset
from torch.utils.data import TensorDataset
from torchaudio.transforms import MFCC
import regex as re
from torch.nn.utils.rnn import pad_sequence

__all__ = ['VietnameseAudio']

class VietnameseAudio(Dataset):
    def __init__(self, root):
        self.root = root

        dataset_dict = {'train': [], 'test': []}
        folder_order = ['train', 'test']
        self.gender_encoder = LabelEncoder()
        
        for subdir in folder_order:
            subdir_path = os.path.join(root, subdir)
            if os.path.isdir(subdir_path):
                prompts_path = os.path.join(subdir_path, 'prompts.txt')
                genders_path = os.path.join(subdir_path, 'genders.txt')

                genders = {}
                with open(genders_path, 'r') as f:
                    for line in f:
                        speaker, gender = line.strip().split()
                        genders[speaker] = gender

                with open(prompts_path, 'r') as f:
                    for line in f:
                        wav_file, transcript = line.strip().split(maxsplit=1)
                        speaker_id = wav_file.split('_')[0]
                        gender = genders.get(speaker_id, 'unknown')
                        wav_path = os.path.join(subdir_path, 'waves', speaker_id, wav_file + '.wav')
                        if os.path.exists(wav_path):
                            transcript = preprocess_text(transcript)
                            data_entry = {'wav_path': wav_path, 'transcript': transcript, 'gender': gender}
                            dataset_dict[subdir].append(data_entry)

                    unique_genders = list(set(genders.values()))
                    self.gender_encoder.fit(unique_genders)
        dataset_dict['train'] = tensor_data(dataset_dict['train'], self.gender_encoder)
        dataset_dict['test'] = tensor_data(dataset_dict['test'], self.gender_encoder)

        super().__init__(train=dataset_dict['train'], test=dataset_dict['test'])
        self.dataset_dict = {'train': dataset_dict['train'], 'test': dataset_dict['test']} 

def tensor_data(data, gender_encoder):
    mfccs = []
    genders = []
    
    for data_entry in data:
        wav_path = data_entry['wav_path']
        gender = data_entry['gender']

        waveform, sample_rate = torchaudio.load(wav_path)

        mfcc_transform = MFCC(sample_rate=sample_rate, n_mfcc=13)
        mfcc = mfcc_transform(waveform)
        padded_mfcc = pad_mfcc(mfcc)
        mfccs.append(padded_mfcc)
        genders.append(gender)

    gender_tensor = torch.tensor(gender_encoder.transform(genders), dtype=torch.long)

    # Create TensorDataset
    tensor_dataset = TensorDataset(torch.stack(mfccs), gender_tensor)

    return tensor_dataset

def preprocess_text(text):
    patterns_to_remove = [
        r'colonsmile', r'colonsad', r'colonsurprise', r'colonlove',
        r'colonsmilesmile', r'coloncontemn', r'colonbigsmile', r'coloncc',
        r'colonsmallsmile', r'coloncolon', r'colonlovelove', r'colonhihi',
        r'doubledot', r'colonsadcolon', r'colonsadcolon', r'colondoublesurprise',
        r'vdotv', r'dotdotdot', r'fraction', r'cshrap'
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def pad_mfcc(mfcc):
    # Ensure the input is 3D (1, n_mfcc, n_frames)
    if mfcc.dim() != 3:
        raise ValueError(f"Expected 3D input for MFCC, but got {mfcc.dim()}D tensor")

    max_length = 100  # Adjust as needed
    n_frames = mfcc.size(2)

    if n_frames < max_length:
        # Padding with zeros
        padding_size = max_length - n_frames
        padded_mfcc = torch.cat((mfcc, torch.zeros(mfcc.size(0), mfcc.size(1), padding_size)), dim=2)
    else:
        # Truncate if necessary
        padded_mfcc = mfcc[:, :, :max_length]

    return padded_mfcc
