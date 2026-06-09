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

def seq2vec(seq: str, max_bits: int, encoding: str = "utf8"):
    if encoding == "utf8":
        raw_bytes = seq.encode('utf-8')
        max_bytes = max_bits // 8
        if len(raw_bytes) > max_bytes:
            raw_bytes = raw_bytes[:max_bytes]
        bits = []
        for b in raw_bytes:
            bits.extend([(b >> i) & 1 for i in range(7, -1, -1)])
        bits += [0] * (max_bits - len(bits))
        chars_used = 0
        byte_pos = 0
        for ch in seq:
            ch_bytes = ch.encode('utf-8')
            if byte_pos + len(ch_bytes) > len(raw_bytes):
                break
            byte_pos += len(ch_bytes)
            chars_used += 1
        return bits, chars_used
    else:
        max_symbols = max_bits // 32
        chars_used = 0
        bits = []
        for ch in seq:
            if chars_used >= max_symbols:
                break
            bits.extend([int(b) for b in format(ord(ch), '032b')])
            chars_used += 1
        bits += [0] * (max_bits - len(bits))
        return bits, chars_used

def vec2seq(vec, encoding: str = "utf8"):
    max_bits = len(vec)
    if encoding == "utf8":
        byte_values = []
        for i in range(0, max_bits, 8):
            byte_val = 0
            for j in range(8):
                if i + j < max_bits and vec[i + j] > 0.5:
                    byte_val |= (1 << (7 - j))
            byte_values.append(byte_val)
        while byte_values and byte_values[-1] == 0:
            byte_values.pop()
        return bytes(byte_values).decode('utf-8', errors='replace')
    else:
        chars = []
        for i in range(0, max_bits, 32):
            bits = vec[i:i+32]
            if any(v > 0.5 for v in bits):
                bits_int = [1 if v > 0.5 else 0 for v in bits]
                chars.append(chr(int(''.join(str(b) for b in bits_int), 2)))
        return ''.join(chars)

def split_into_chunks(text: str, max_bits: int, encoding: str = "utf8"):
    chunks = []
    i = 0
    if encoding == "utf8":
        while i < len(text):
            bits, used = seq2vec(text[i:], max_bits, encoding)
            chunks.append((text[i:i+used], bits))
            i += used
    else:
        max_symbols = max_bits // 32
        while i < len(text):
            chunk_text = text[i:i+max_symbols]
            bits, used = seq2vec(chunk_text, max_bits, encoding)
            chunks.append((chunk_text, bits))
            i += used
    return chunks

class TextBitDataset(Dataset):
    def __init__(self, text: str, config):
        self.config = config
        self.encoding = config.encoding
        max_bits = config.seq_len * 32
        cache_dir = os.path.join("data", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"dataset_{config.seq_len}_{config.encoding}.pt")

        if os.path.exists(cache_file):
            self.data = torch.load(cache_file)
            return

        if self.encoding == "utf8":
            byte_data = text.encode('utf-8')
            byte_array = np.frombuffer(byte_data, dtype=np.uint8)
            bits_array = np.unpackbits(byte_array)
        else:
            codes = np.array([ord(ch) for ch in text], dtype=np.uint32)
            byte_view = codes.view(dtype=np.uint8)
            bits_array = np.unpackbits(byte_view)

        total_bits = len(bits_array)
        pad = (max_bits - total_bits % max_bits) % max_bits
        if pad:
            bits_array = np.pad(bits_array, (0, pad), constant_values=0)
        chunks = bits_array.reshape(-1, max_bits).astype(np.float32)
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