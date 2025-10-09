from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from typing import List


def train_bpe_tokenizer(words_file: str, vocab_size: int, save_path: str):
    """Train a BPE tokenizer on the provided words file and save it to save_path."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[MASK]"])
    tokenizer.train([words_file], trainer)
    tokenizer.save(save_path)
    print(f"Saved trained BPE tokenizer to {save_path}")
    return save_path

def load_tokenizer(path: str) -> Tokenizer:
    tok = Tokenizer.from_file(path)
    return tok

def load_words_file(path: str) -> List[str]:
    with open(path, 'r', encoding='utf8') as f:
        words = [w.strip() for w in f if w.strip()]
    seen = set()
    uniq = []
    for w in words:
        if w not in seen:
            uniq.append(w)
            seen.add(w)
    return uniq


def encode_words_to_ids(words: List[str], tokenizer: Tokenizer, max_len: int, pad_id: int):
    """Encode each word into a fixed-length sequence of token ids (pad/truncate)."""
    encodings = []
    for w in words:
        enc = tokenizer.encode(w)
        ids = enc.ids
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [pad_id] * (max_len - len(ids))
        encodings.append(ids)
    return encodings

