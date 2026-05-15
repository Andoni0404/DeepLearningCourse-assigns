"""
models.py — Universal Sequence Lab
All sequential architectures implemented in PyTorch.
Covers: VanillaRNN, LSTM, GRU, Transformer (Encoder-only & Encoder-Decoder)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. VANILLA RNN (baseline — shows limitations)
# ─────────────────────────────────────────────
class VanillaRNN(nn.Module):
    """Simple RNN. Used as baseline to demonstrate vanishing gradient."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim,
                 n_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embedded)
        # Use last hidden state
        out = self.fc(self.dropout(hidden[-1]))
        return out


# ─────────────────────────────────────────────
# 2. LSTM CLASSIFIER
# ─────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    """Bidirectional LSTM for sequence classification."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim,
                 n_layers=2, bidirectional=True, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        embedded = self.dropout(self.embedding(x))
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(embedded)
        # Concat forward + backward last hidden
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        return self.fc(self.dropout(hidden))


# ─────────────────────────────────────────────
# 3. GRU CLASSIFIER
# ─────────────────────────────────────────────
class GRUClassifier(nn.Module):
    """Bidirectional GRU for sequence classification."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim,
                 n_layers=2, bidirectional=True, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers,
                          batch_first=True, bidirectional=bidirectional,
                          dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        embedded = self.dropout(self.embedding(x))
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.gru(packed)
        else:
            _, hidden = self.gru(embedded)
        if self.gru.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        return self.fc(self.dropout(hidden))


# ─────────────────────────────────────────────
# 4. TRANSFORMER CLASSIFIER (Encoder only)
# ─────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """Transformer encoder + CLS token pooling for classification."""
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, output_dim,
                 num_layers=3, dropout=0.1, pad_idx=0, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(embed_dim, dropout, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx

    def forward(self, x, lengths=None):
        # Build padding mask
        src_key_padding_mask = (x == self.pad_idx)
        embedded = self.pos_enc(self.embedding(x))
        out = self.transformer(embedded, src_key_padding_mask=src_key_padding_mask)
        # Mean pooling over non-padding tokens
        mask = (~src_key_padding_mask).unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.fc(self.dropout(pooled))


# ─────────────────────────────────────────────
# 5. LSTM FORECASTER (Time Series)
# ─────────────────────────────────────────────
class LSTMForecaster(nn.Module):
    """Multi-layer LSTM for univariate/multivariate time series forecasting."""
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1]))


class GRUForecaster(nn.Module):
    """Multi-layer GRU for time series forecasting."""
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.dropout(out[:, -1]))


class TransformerForecaster(nn.Module):
    """Transformer encoder for time series forecasting."""
    def __init__(self, input_dim, d_model, num_heads, ff_dim, output_dim,
                 num_layers=2, dropout=0.1, max_len=512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pos_enc(self.input_proj(x))
        out = self.transformer(x)
        return self.fc(self.dropout(out[:, -1]))


# ─────────────────────────────────────────────
# 6. SEQ2SEQ with ATTENTION (Translation)
# ─────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention mechanism."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch, hidden), encoder_outputs: (batch, src_len, hidden)
        src_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(hidden_dim)
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_dim * 2 + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_token, hidden, cell, encoder_outputs):
        tgt_token = tgt_token.unsqueeze(1)  # (batch, 1)
        embedded = self.dropout(self.embedding(tgt_token))  # (batch, 1, embed)
        # Attention over last layer of hidden
        attn_weights = self.attention(hidden[-1], encoder_outputs)  # (batch, src_len)
        attn_weights = attn_weights.unsqueeze(1)  # (batch, 1, src_len)
        context = torch.bmm(attn_weights, encoder_outputs)  # (batch, 1, hidden)
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        pred = self.fc_out(torch.cat((output, context, embedded), dim=2).squeeze(1))
        return pred, hidden, cell, attn_weights.squeeze(1)


class Seq2SeqTranslator(nn.Module):
    """Full Seq2Seq model with Bahdanau attention for translation."""
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.size()
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)
        attention_maps = []

        encoder_outputs, hidden, cell = self.encoder(src)
        dec_input = tgt[:, 0]

        for t in range(1, tgt_len):
            pred, hidden, cell, attn = self.decoder(
                dec_input, hidden, cell, encoder_outputs)
            outputs[:, t] = pred
            attention_maps.append(attn.detach().cpu())
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            dec_input = tgt[:, t] if teacher_force else pred.argmax(1)

        return outputs, attention_maps
