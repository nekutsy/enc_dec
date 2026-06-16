"""Quick inference test — best models from each seq_len on random samples."""

import torch, sys, os, random
sys.path.insert(0, os.path.dirname(__file__))

from configs import UNICODE_BITS
from model import Autoencoder
from data import _build_full_bits, load_text, vec2seq, chars_to_bits
import numpy as np

device = torch.device("cpu")  # inference only — no compile overhead

BEST = {2: 12, 4: 14, 8: 9, 16: 8, 32: 6, 64: 3, 128: 1}

def _load_model(seq_len, n_hidden):
    input_dim = seq_len * UNICODE_BITS
    hidden_dim = 4 * input_dim
    bottleneck = seq_len
    layer_sizes = [input_dim] + [hidden_dim] * n_hidden + [bottleneck] + [hidden_dim] * n_hidden + [input_dim]
    key = "_".join(map(str, layer_sizes))
    model_name = f"sweep_s{seq_len}_h{n_hidden}"
    path = f"sessions/sweep/{key}_{model_name}.pth"
    model = Autoencoder(layer_sizes).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    if any(k.startswith('_orig_mod.') for k in state.keys()):
        state = {k[len('_orig_mod.'):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model

def main():
    text = load_text()
    full_bits = _build_full_bits(text)
    n_chars = full_bits.numel() // UNICODE_BITS

    # Pick 3 random positions and extract gold text of max seq_len
    rng = random.Random(42)
    positions = [rng.randint(0, n_chars - 128) for _ in range(3)]
    samples = []
    for pos in positions:
        gold = text[pos:pos+128]
        samples.append((pos, gold))
        print(f"  Sample @{pos}: «{gold[:60]}...»")

    print()

    with torch.inference_mode():
        for sl, nh in sorted(BEST.items()):
            model = _load_model(sl, nh)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"\n── seq_len={sl}  h={nh}  ({n_params:,} params) ──", flush=True)

            for pos, gold in samples:
                # Encode exactly seq_len chars as bits
                chunk_chars = gold[:sl]
                codes = np.array([ord(ch) for ch in chunk_chars], dtype=np.uint32)
                bits_np = chars_to_bits(codes).ravel()
                inp = torch.from_numpy(bits_np).float().unsqueeze(0).to(device)
                out = model(inp).squeeze(0).cpu().numpy()
                rec = vec2seq(out)
                errors = sum(1 for a, b in zip(chunk_chars, rec) if a != b)
                err_bits = np.abs(bits_np - out).sum()
                print(f"  @{pos}: orig={chunk_chars!r} → rec={rec[:len(chunk_chars)]!r}  char_err={errors}/{sl}  bit_err={err_bits:.1f}")

    print("\n✓ done")

if __name__ == "__main__":
    main()
