"""
train.py — Universal Sequence Lab
Generic training loop, evaluation, and utilities reusable across all modules.
"""

import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# ─────────────────────────────────────────────
# TRAINING & EVALUATION LOOPS
# ─────────────────────────────────────────────

def train_epoch(model, iterator, optimizer, criterion, device, clip=1.0,
                task='classification', use_lengths=False):
    model.train()
    epoch_loss = 0

    for batch in iterator:
        optimizer.zero_grad()

        if task == 'classification':
            if use_lengths:
                src, lengths, labels = batch
                src, labels = src.to(device), labels.to(device)
                pred = model(src, lengths)
            else:
                src, labels = batch
                src, labels = src.to(device), labels.to(device)
                pred = model(src)
            loss = criterion(pred, labels)

        elif task == 'forecasting':
            x, y = batch
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)

        elif task == 'seq2seq':
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)
            pred, _ = model(src, tgt)
            # pred: (batch, tgt_len, vocab), tgt: (batch, tgt_len)
            pred = pred[:, 1:].reshape(-1, pred.shape[-1])
            tgt = tgt[:, 1:].reshape(-1)
            loss = criterion(pred, tgt)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device, task='classification',
             use_lengths=False):
    model.eval()
    epoch_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in iterator:
            if task == 'classification':
                if use_lengths:
                    src, lengths, labels = batch
                    src, labels = src.to(device), labels.to(device)
                    pred = model(src, lengths)
                else:
                    src, labels = batch
                    src, labels = src.to(device), labels.to(device)
                    pred = model(src)
                loss = criterion(pred, labels)
                all_preds.extend(pred.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            elif task == 'forecasting':
                x, y = batch
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

            elif task == 'seq2seq':
                src, tgt = batch
                src, tgt = src.to(device), tgt.to(device)
                pred, _ = model(src, tgt, teacher_forcing_ratio=0.0)
                pred_flat = pred[:, 1:].reshape(-1, pred.shape[-1])
                tgt_flat = tgt[:, 1:].reshape(-1)
                loss = criterion(pred_flat, tgt_flat)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator), np.array(all_preds), np.array(all_labels)


def train_model(model, train_iter, val_iter, optimizer, criterion, device,
                n_epochs=10, task='classification', use_lengths=False,
                scheduler=None, model_name='model', save_path='models/best.pt'):
    """Full training loop with early stopping and metric tracking."""
    history = {'train_loss': [], 'val_loss': [], 'val_metric': []}
    best_val_loss = float('inf')

    for epoch in range(1, n_epochs + 1):
        start = time.time()
        train_loss = train_epoch(model, train_iter, optimizer, criterion, device,
                                 task=task, use_lengths=use_lengths)
        val_loss, preds, labels = evaluate(model, val_iter, criterion, device,
                                           task=task, use_lengths=use_lengths)
        if scheduler:
            scheduler.step(val_loss)

        # Compute metric
        if task == 'classification':
            metric = (preds == labels).mean()
            metric_name = 'Acc'
        elif task == 'forecasting':
            metric = np.sqrt(np.mean((preds - labels) ** 2))
            metric_name = 'RMSE'
        else:
            metric = val_loss  # Use BLEU externally
            metric_name = 'Loss'

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metric'].append(metric)

        elapsed = time.time() - start
        print(f"[{model_name}] Epoch {epoch:02d}/{n_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val {metric_name}: {metric:.4f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    return history


# ─────────────────────────────────────────────
# VISUALIZATION UTILITIES
# ─────────────────────────────────────────────

def plot_learning_curves(histories: dict, task='classification', figsize=(14, 5)):
    """Plot train/val loss and metric curves for multiple models."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    metric_name = 'Accuracy' if task == 'classification' else \
                  'RMSE' if task == 'forecasting' else 'Val Loss'

    for name, hist in histories.items():
        axes[0].plot(hist['train_loss'], label=f'{name} train')
        axes[0].plot(hist['val_loss'], '--', label=f'{name} val')
        axes[1].plot(hist['val_metric'], label=name)

    axes[0].set_title('Loss Curves', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title(f'Validation {metric_name}', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel(metric_name)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.show()
    return fig


def plot_attention_heatmap(attn_weights, src_tokens, tgt_tokens,
                           title='Attention Weights'):
    """Visualize attention weights for Seq2Seq model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    attn = attn_weights.numpy() if hasattr(attn_weights, 'numpy') else attn_weights
    sns.heatmap(attn, xticklabels=src_tokens, yticklabels=tgt_tokens,
                cmap='viridis', ax=ax)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Source tokens')
    ax.set_ylabel('Target tokens')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    return fig


def plot_forecast(y_true, y_pred, title='Forecast vs Actual', n_samples=200):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y_true[:n_samples], label='Actual', linewidth=1.5)
    ax.plot(y_pred[:n_samples], '--', label='Predicted', linewidth=1.5)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")
    return trainable


def print_classification_report(y_true, y_pred, target_names):
    print(classification_report(y_true, y_pred, target_names=target_names))


# ─────────────────────────────────────────────
# BLEU SCORE (for translation)
# ─────────────────────────────────────────────

def compute_bleu(model, iterator, tgt_vocab, device, max_samples=500):
    """Compute corpus BLEU score using sacrebleu."""
    try:
        import sacrebleu
    except ImportError:
        print("sacrebleu not installed. Run: pip install sacrebleu")
        return None

    model.eval()
    hypotheses, references = [], []
    eos_idx = tgt_vocab['<eos>']
    pad_idx = tgt_vocab['<pad>']

    with torch.no_grad():
        for i, (src, tgt) in enumerate(iterator):
            if i * src.size(0) > max_samples:
                break
            src = src.to(device)
            pred, _ = model(src, tgt.to(device), teacher_forcing_ratio=0.0)
            pred_ids = pred.argmax(-1).cpu()
            inv_vocab = {v: k for k, v in tgt_vocab.items()}

            for j in range(src.size(0)):
                hyp_tokens, ref_tokens = [], []
                for idx in pred_ids[j, 1:]:
                    if idx.item() in (eos_idx, pad_idx):
                        break
                    hyp_tokens.append(inv_vocab.get(idx.item(), '<unk>'))
                for idx in tgt[j, 1:]:
                    if idx.item() in (eos_idx, pad_idx):
                        break
                    ref_tokens.append(inv_vocab.get(idx.item(), '<unk>'))
                hypotheses.append(' '.join(hyp_tokens))
                references.append(' '.join(ref_tokens))

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score
