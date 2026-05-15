"""
dataset.py — Universal Sequence Lab
Dataset loading, preprocessing, and DataLoader factories for all 3 modules.
  - Module A: IMDb (sentiment classification)
  - Module B: Weather / Financial time series (forecasting)
  - Module C: Multi30k EN→DE (translation)
"""

import re
import torch
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ══════════════════════════════════════════════
# MODULE A — NLP: IMDb Sentiment
# ══════════════════════════════════════════════

def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)          # strip HTML
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.split()


class Vocabulary:
    def __init__(self, min_freq=2, max_size=25_000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.token2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        self.idx2token = {v: k for k, v in self.token2idx.items()}

    def build(self, texts, tokenizer=simple_tokenize):
        counter = Counter()
        for text in texts:
            counter.update(tokenizer(text))
        for token, freq in counter.most_common(self.max_size):
            if freq >= self.min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
        print(f"  Vocabulary size: {len(self.token2idx):,}")
        return self

    def encode(self, text, tokenizer=simple_tokenize, max_len=256):
        tokens = tokenizer(text)[:max_len]
        return [self.token2idx.get(t, 1) for t in tokens]  # 1 = <unk>

    def __len__(self):
        return len(self.token2idx)

    def __getitem__(self, key):
        return self.token2idx[key]


class IMDbDataset(Dataset):
    """HuggingFace IMDb dataset wrapper."""
    def __init__(self, hf_dataset, vocab, max_len=256):
        self.data = hf_dataset
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']
        ids = self.vocab.encode(text, max_len=self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def collate_nlp(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(t) for t in texts])
    padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded, lengths, labels


def get_imdb_loaders(batch_size=64, max_len=256, min_freq=3):
    """Returns (train_loader, val_loader, test_loader, vocab)."""
    from datasets import load_dataset
    print("Loading IMDb dataset...")
    ds = load_dataset('imdb')

    # Build vocab from train split only
    print("Building vocabulary...")
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build([ex['text'] for ex in ds['train']])

    train_ds = IMDbDataset(ds['train'], vocab, max_len)
    test_ds  = IMDbDataset(ds['test'],  vocab, max_len)

    # Split train into train/val (90/10)
    n_val = int(0.1 * len(train_ds))
    train_sub, val_sub = torch.utils.data.random_split(
        train_ds, [len(train_ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_sub, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_nlp)
    val_loader   = DataLoader(val_sub,   batch_size=batch_size,
                              collate_fn=collate_nlp)
    test_loader  = DataLoader(test_ds,   batch_size=batch_size,
                              collate_fn=collate_nlp)
    print(f"  Train: {len(train_sub)} | Val: {len(val_sub)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, vocab


# ══════════════════════════════════════════════
# MODULE B — Time Series: Weather Forecasting
# ══════════════════════════════════════════════

class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset for time series forecasting.
    Input:  window of `seq_len` steps  → shape (seq_len, n_features)
    Target: next `horizon` steps       → shape (horizon,)  [first feature only]
    """
    def __init__(self, data: np.ndarray, seq_len=96, horizon=24):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.horizon, 0]
        return x, y


def get_weather_loaders(batch_size=64, seq_len=96, horizon=24, val_ratio=0.1, test_ratio=0.2):
    """
    Downloads the Jena Climate dataset (temp, pressure, humidity, etc.)
    and returns (train_loader, val_loader, test_loader, scaler).
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
    print("Downloading Jena Climate dataset...")
    df = pd.read_csv(url, compression='zip')
    df = df.iloc[::6]  # Downsample to hourly

    features = ['T (degC)', 'p (mbar)', 'rh (%)', 'wv (m/s)', 'Tdew (degC)']
    data = df[features].values.astype(np.float32)

    # Normalize
    n_train = int(len(data) * (1 - val_ratio - test_ratio))
    n_val   = int(len(data) * val_ratio)
    scaler = StandardScaler()
    data[:n_train] = scaler.fit_transform(data[:n_train])
    data[n_train:]  = scaler.transform(data[n_train:])

    train_ds = TimeSeriesDataset(data[:n_train], seq_len, horizon)
    val_ds   = TimeSeriesDataset(data[n_train:n_train + n_val], seq_len, horizon)
    test_ds  = TimeSeriesDataset(data[n_train + n_val:], seq_len, horizon)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    print(f"  Features: {features}")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, scaler, features


# ══════════════════════════════════════════════
# MODULE C — Seq2Seq: Translation EN→DE
# ══════════════════════════════════════════════

class TranslationVocabulary(Vocabulary):
    def build_from_pairs(self, sentence_list, tokenizer=str.split):
        counter = Counter()
        for sent in sentence_list:
            counter.update(tokenizer(sent.lower()))
        for token, freq in counter.most_common(self.max_size):
            if freq >= self.min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[idx] = token  # NOTE: intentionally inverted for compat
                self.token2idx[token] = idx
                self.idx2token[idx] = token
        return self

    def encode_with_eos(self, text, max_len=50):
        tokens = text.lower().split()[:max_len]
        ids = [self.token2idx.get(t, 1) for t in tokens]
        return [self['<sos>']] + ids + [self['<eos>']]


class TranslationDataset(Dataset):
    def __init__(self, src_sents, tgt_sents, src_vocab, tgt_vocab, max_len=50):
        self.pairs = list(zip(src_sents, tgt_sents))
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_sent, tgt_sent = self.pairs[idx]
        src_ids = self.src_vocab.encode_with_eos(src_sent, self.max_len)
        tgt_ids = self.tgt_vocab.encode_with_eos(tgt_sent, self.max_len)
        return torch.tensor(src_ids, dtype=torch.long), \
               torch.tensor(tgt_ids, dtype=torch.long)


def collate_seq2seq(batch):
    src, tgt = zip(*batch)
    src_padded = pad_sequence(src, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt, batch_first=True, padding_value=0)
    return src_padded, tgt_padded


def get_translation_loaders(batch_size=128, max_len=50, min_freq=2):
    """
    Returns (train_loader, val_loader, test_loader, src_vocab, tgt_vocab)
    using the Multi30k EN→DE dataset from HuggingFace.
    """
    from datasets import load_dataset
    print("Loading Multi30k EN→DE dataset...")
    ds = load_dataset('bentrevett/multi30k')

    src_train = [ex['en'] for ex in ds['train']]
    tgt_train = [ex['de'] for ex in ds['train']]

    print("Building source (EN) vocabulary...")
    src_vocab = TranslationVocabulary(min_freq=min_freq)
    src_vocab.build(src_train, tokenizer=str.split)

    print("Building target (DE) vocabulary...")
    tgt_vocab = TranslationVocabulary(min_freq=min_freq)
    tgt_vocab.build(tgt_train, tokenizer=str.split)

    def make_loader(split, shuffle=False):
        src_sents = [ex['en'] for ex in ds[split]]
        tgt_sents = [ex['de'] for ex in ds[split]]
        dataset = TranslationDataset(src_sents, tgt_sents,
                                     src_vocab, tgt_vocab, max_len)
        return DataLoader(dataset, batch_size=batch_size,
                          shuffle=shuffle, collate_fn=collate_seq2seq)

    train_loader = make_loader('train', shuffle=True)
    val_loader   = make_loader('validation')
    test_loader  = make_loader('test')

    print(f"  EN vocab: {len(src_vocab):,} | DE vocab: {len(tgt_vocab):,}")
    print(f"  Train: {len(ds['train'])} | Val: {len(ds['validation'])} | Test: {len(ds['test'])}")
    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
