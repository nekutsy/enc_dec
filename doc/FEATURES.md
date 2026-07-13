# FEATURES.md — Фичи: VAE, denoising, fine-tune, асимметрия

> CLI-команды → [USAGE.md](USAGE.md)
> Конфигурация → [CONFIG.md](CONFIG.md)

---

## VAE-режим (Variational Autoencoder)

Энкодер выдаёт μ и log σ², декодер обучается на семплах из N(μ, σ²).

```bash
# Стандартный VAE
bin/enc-dec train --n 6 --budget 384M --samples 50M --vae

# β-VAE — управление балансом KL vs реконструкция
bin/enc-dec train --n 6 --budget 160M --samples 50M --vae --vae-beta 0.1
```

**Архитектурные отличия от обычного AE:**
- Два дополнительных линейных слоя на bottleneck для μ и log σ²
- Число параметров почти не меняется (μ и log σ² той же размерности)
- `forward()` возвращает `(reconstruction, μ, log σ²)`
- `encode()` возвращает μ (детерминированно)
- `model.sample(N)` — генерация из N(0,I) приора

**Loss:** `BCE(recon, target) + β · KL(μ, σ² || N(0,I))`

### β-балансировка

| β | Эффект |
|---|--------|
| 0 | Обычный автоэнкодер (KL отключён) |
| < 1 | Ослабленная регуляризация, лучше реконструкция, хуже генерация |
| 1 | Стандартный VAE |
| > 1 | Усиленная регуляризация, лучше диспентанглинг |

### Инференс VAE

```
> 0                    # загрузить модель
> sample               # 1 сэмпл из приора
> sample 5             # 5 сэмплов
> info                 # mode: VAE  β=1.0
```

### Sweep VAE

```bash
# Sweep по β
bin/enc-dec sweep grid --vary noise_prob=0.0,0.25 --fixed n=6 \
  --solve b --budget 40M --override model.vae=true training.vae_beta=0.5
```

Готовые конфиги: `configs/vae_*.json`.

---

## Denoising (шум)

Модель обучается как denoising autoencoder: вход зашумляется, target — чистый. Шум добавляется на уровне uint21-значений: с вероятностью `noise_prob` к значению символа добавляется N(0, noise_std²), затем округление и clamp в [0, 2²¹−1].

### Фиксированный шум

```bash
bin/enc-dec train --n 3 --budget 160M --samples 50M --noise-prob 0.25
# --noise-prob 0.25 ⇔ --noise-prob-min 0.25 --noise-prob-max 0.25
```

### Per-batch сэмплинг

`noise_prob` и `noise_std` не фиксированы, а берутся из диапазона `[min, max]` для каждого семпла.

Две стратегии сэмплинга:

| Стратегия | Поведение |
|-----------|----------|
| `linear` (default) | `noise = min + (max - min) · (idx % stride) / (stride - 1)`. Каждый батч покрывает весь диапазон. Детерминированно. |
| `uniform` | Случайный `uniform(min, max)` для каждого семпла. |

Семплы с `noise_prob ≤ 0` сразу возвращаются без шума — при `min=0` часть батча всегда чистая.

```bash
# Диапазон [0, 0.5] с линейной интерполяцией
bin/enc-dec train --noise-prob-min 0.0 --noise-prob-max 0.5

# Uniform random
bin/enc-dec train --noise-prob-min 0.0 --noise-prob-max 0.5 --noise-strategy uniform

# Варьировать и prob, и std одновременно
bin/enc-dec train --noise-prob-min 0.0 --noise-prob-max 0.3 \
  --noise-std-min 1.0 --noise-std-max 5.0

# Кастомный stride
bin/enc-dec train --noise-prob-min 0.0 --noise-prob-max 0.5 \
  --noise-strategy linear --noise-stride 128
```

---

## Fine-tune (`--pretrain-from`)

Начать обучение с весов существующего рана вместо случайной инициализации.

```bash
bin/enc-dec train --pretrain-from cce656d8f25a --samples 50M --lr 0.0001

# Другой шедулер, больше семплов, с шумом
bin/enc-dec train --pretrain-from cce656d8f25a --samples 100M \
  --lr 0.00005 --scheduler greedy --noise-prob 0.25
```

**Что происходит:**
- **Архитектура** наследуется от донора (из `meta.json`) — `--n`, `--b`, `--seq-len`, `--shape` игнорируются
- **Веса** грузятся из `model.pth` (или `best.pth`) донора
- **Оптимизатор/шедулер** — с нуля (чистый fine-tune, моментумы не копируются)
- **TrainConfig** задаётся CLI-аргументами как обычно
- **Донор не модифицируется** — новый run_id, своя директория, свой лог

**Дедупликация:** `pretrain_run_id` входит в `training_hash` → архитектуры с разным претрейном — разные раны.

---

## Асимметричные архитектуры

Энкодер и декодер могут иметь разное количество слоёв. Бюджет параметров распределяется между ними: чем больше слоёв, тем ýже каждый.

```bash
# Энкодер 8 слоёв, декодер 3 слоя
bin/enc-dec train --enc-n 8 --dec-n 3 --budget 40M --samples 5M

# Sweep по комбинациям
bin/enc-dec sweep grid --vary enc_n=3,5,7 --fixed dec_n=2 --solve b --budget 40M

# Overfit
bin/enc-dec overfit --seq-len 32 --enc-n 4 --dec-n 1 --b 1.5

# LR finder
bin/enc-dec lr-find --seq-len 64 --enc-n 6 --dec-n 2 --budget 40M
```

**Shorthand:**
- `--n` — симметрично, задаёт и encoder и decoder
- `--enc-n` / `--dec-n` — точный контроль, переопределяет `--n`
- В JSON-конфигах: поля `enc_n` / `dec_n` в `model.*`
- `sweep.fixed` поддерживает `enc_n`/`dec_n` наравне с `n`

Готовые конфиги: `configs/asym_rect_*.json`, `configs/enc_dec_ratio_sweep_n*.json`.
