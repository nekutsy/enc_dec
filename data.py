import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

def load_text(data_dir="data/dataset"):
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    texts = []
    if not txt_files:
        print(f"No .txt files found in {data_dir}, using dummy text.")
        dummy = "Это тестовый текст для автоэнкодера. " * 50
        texts.append(dummy)
    else:
        print(f"Found {len(txt_files)} .txt file(s) in {data_dir}:")
        for path in txt_files:
            filename = os.path.basename(path)
            with open(path, "r", encoding="utf8") as f:
                content = f.read()
                texts.append(content)
                print(f"  - {filename}: {len(content)} characters")
    return "".join(texts)

def seq2vec(seq: str, max_bits: int):
    max_symbols = max_bits // 21
    chars_used = 0
    bits = []
    for ch in seq:
        if chars_used >= max_symbols:
            break
        codepoint = ord(ch)
        bits.extend([(codepoint >> (20 - i)) & 1 for i in range(21)])
        chars_used += 1
    bits += [0] * (max_bits - len(bits))
    return bits, chars_used

def vec2seq(vec):
    arr = np.array(vec, dtype=np.float32).reshape(-1, 21)
    powers = 2 ** np.arange(20, -1, -1)
    codepoints = ((arr > 0.5) @ powers).astype(int)
    valid_codes = codepoints[codepoints > 0]
    return ''.join(chr(c) for c in valid_codes)

def split_into_chunks(text: str, max_bits: int):
    chunks = []
    i = 0
    max_symbols = max_bits // 21
    while i < len(text):
        chunk_text = text[i:i+max_symbols]
        bits, used = seq2vec(chunk_text, max_bits)
        chunks.append((chunk_text, bits))
        i += used
    return chunks

class TextBitDataset(Dataset):
    def __init__(self, text: str, config):
        self.config = config
        max_bits = config.seq_len * 21
        cache_dir = os.path.join("data", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"dataset_{config.seq_len}_unicode21.pt")

        if os.path.exists(cache_file):
            self.data = torch.load(cache_file)
            return

        codes = np.array([ord(ch) for ch in text], dtype=np.uint32)
        bits = np.zeros((len(codes), 21), dtype=np.float32)
        for i in range(21):
            bits[:, i] = (codes >> (20 - i)) & 1

        bits_array = bits.reshape(-1)
        total_bits = len(bits_array)
        pad = (max_bits - total_bits % max_bits) % max_bits
        if pad:
            bits_array = np.pad(bits_array, (0, pad), constant_values=0)
        chunks = bits_array.reshape(-1, max_bits)
        self.data = torch.from_numpy(chunks)
        torch.save(self.data, cache_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def prepare_data(text: str, config):
    dataset = TextBitDataset(text, config)
    data = dataset.data
    indices = torch.randperm(len(data))
    train_size = int(len(data) * config.train_ratio)
    return data[indices[:train_size]], data[indices[train_size:]]

def export_latent_vectors(model, text, config, device, output_path="data/latent/latent_vectors.pt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    dataset = TextBitDataset(text, config)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    latents = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = model.encode(batch)
            latents.append(z.cpu())
    latents = torch.cat(latents, dim=0)
    torch.save(latents, output_path)
    print(f"Exported latent vectors: {latents.shape} -> {output_path}")

def load_latent_vectors(path="data/latent/latent_vectors.pt"):
    return torch.load(path, map_location='cpu')