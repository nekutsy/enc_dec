# DATA.md — Данные, кодировка, пайплайн

> Релевантные файлы: `encoding/unicode21.py`, `data/core.py`, `data/pipeline.py`

## Кодировка Unicode-21

Каждый символ ∈ [U+0000, U+1FFFFF] → 21 бит (MSB first).

Реализация: `encoding/unicode21.py` — функции `encode_char()`, `decode_bits()`.

## SlidingWindowDataset (`data/core.py`)

Окна со **stride=1 символ** — каждый символ появляется в `seq_len` окнах.

Реализован через `torch.as_strided` на float32-тензоре, без копирования данных. Фильтрация: только окна ≥1 реального символа (null-padding gaps пропускаются).

```python
# Каждое окно: seq_len × 21 входных бит → цель: те же 21 бит
dataset[i] → (window_bits, target_bits)
```

## Двухуровневое кэширование (`data/pipeline.py`)

| Уровень | Хранение | Размер (~50M символов) |
|---------|----------|------------------------|
| Диск | uint8 packed (`data/cache/full_bits.u8`) | ~0.13 GB |
| RAM | float32 unpacked | ~4.2 GB |

Pipeline:
1. Чтение `.txt` из `data/dataset/`
2. `unicode21.encode_char(ch)` для каждого символа → 21 бит
3. Pack в uint8 → `full_bits.u8` (дисковый кэш)
4. При загрузке: unpack в float32 → тензор в RAM
5. `SlidingWindowDataset` создаёт окна через `as_strided`

## Данные

`data/dataset/` — русская классика:
- Проза: Чехов
- Поэзия: Пушкин, Блок, Лермонтов, Некрасов

~50M символов. Формат: `.txt` файлы.

## NoisyDataset

Wrapper для denoising: добавляет Gaussian шум на уровне uint21-значений:

1. С вероятностью `noise_prob`: `value += N(0, noise_std²)`
2. Округление до целого
3. Clamp в `[0, 2²¹−1]`

Per-batch сэмплинг: `noise_prob` и `noise_std` берутся из диапазона `[min, max]` для каждого семпла (strategies: `linear`, `uniform`).

## Ключевые модули

| Файл | Ответственность |
|------|----------------|
| `encoding/unicode21.py` | `encode_char()`, `decode_bits()` |
| `data/core.py` | `SlidingWindowDataset`, `NoisyDataset` |
| `data/pipeline.py` | Кэширование, загрузка, пайплайн |
