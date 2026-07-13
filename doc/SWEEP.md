# SWEEP.md — Стратегии sweep

> CLI-команды sweep → [USAGE.md](USAGE.md)
> JSON-формат конфигов → [CONFIG.md](CONFIG.md)

Sweep-система перебирает параметры архитектуры/обучения по заданной стратегии. Registry (SQLite) обеспечивает дедупликацию: если модель уже обучена — пропускает, если частично — возобновляет.

## Команды

```bash
# Из JSON-конфига (основной способ)
bin/enc-dec sweep run --config configs/noise_sweep.json

# С оверрайдом
bin/enc-dec sweep run --config configs/rect_sweep_384m.json \
  --override model.seq_len=64 sweep.strategy=binary

# Shorthand из CLI (без JSON)
bin/enc-dec sweep grid --vary n=2,4,6,8 --solve b --budget 40M
bin/enc-dec sweep binary --vary n --range 2 16 --solve b --budget 40M
```

---

## Стратегии

### `grid` — полный перебор

```json
"sweep": {
  "strategy": "grid",
  "vary": "n",
  "values": [2, 4, 6, 8, 10],
  "solve": "b",
  "budget": 40000000
}
```

Перебирает все `values` по порядку. С `solve: "b"` для каждого n подбирает коэффициент ширины `b` под заданный бюджет.

### `binary` — бинарный поиск

```json
"sweep": {
  "strategy": "binary",
  "vary": "n",
  "values": [2, 16],
  "solve": "b",
  "budget": 40000000
}
```

`values` — границы `[lo, hi]`. Алгоритм:
1. Пробует границы (lo, hi)
2. Сравнивает loss — выбирает лучшую
3. Биссектирует между лучшей и второй лучшей
4. Повторяет до сходимости (соседние значения проверены)

### `grid` с вложенным бинарным поиском

```json
"sweep": {
  "strategy": "grid",
  "vary": "noise_prob",
  "values": [0.01, 0.05, 0.1, 0.25, 0.5, 0.75],
  "binary_on": "lr",
  "binary_range": [0.0001, 0.01],
  "fixed": {"n": 6},
  "solve": "b",
  "budget": 384000000
}
```

Каждый `noise_prob` → бинарный поиск оптимального `lr` → запись лучшего.

---

## Параметр `solve`

Управляет тем, как заполняются свободные параметры архитектуры:

| solve | Описание |
|-------|----------|
| `"b"` | По `n` + `budget` подбирает коэффициент ширины `b` |
| `"n"` | По `b` + `budget` подбирает количество слоёв `n` |
| `null` | Оба заданы вручную (через `fixed` или values) |

---

## Параметр `vary`

Что варьируется при sweep'е:

**Архитектурные** (`MODEL_LEVEL_VARY`):
`n`, `enc_n`, `dec_n`, `b`, `bottleneck`, `normalization`, `activation`, `dropout`, `norm_bottleneck`, `norm_last`, `trapezoid_alpha`, `residual`, `residual_norm`

**Тренировочные** (`TRAIN_LEVEL_VARY`):
`lr`, `scheduler`, `grad_clip`, `optimizer`, `weight_decay`, `batch_size`, `num_workers`, `noise_prob`, `noise_std`, `noise_prob_min`, `noise_prob_max`

Если `vary` — training-level параметр, в `fixed` должны быть заданы `n` и/или `b`.

### Per-batch шум в sweep

Скалярные значения (0.0, 0.25) — фиксированный `min=max`. Список `[0.0, 0.5]` — диапазон, `linear` стратегия по умолчанию:

```json
"sweep": {
  "vary": "noise_prob_min",
  "values": [0.0, 0.25, 0.5, [0.0, 0.5]]
}
```

---

## Архитектурные формы в sweep

Задаются через `model.shape`:

| `shape` | Описание |
|---------|----------|
| `rectangular` | `[D] + [H]×n + [B] + [H]×n + [D]` — одинаковая ширина |
| `pyramid` | Сужающаяся к центру |
| `interleaved` | Wide-слои чередуются с narrow |
| `trapezoid` | Линейная интерполяция + α-отклонение |

---

## Дедупликация и кэширование

Registry (SQLite `sessions/registry.db`) хранит fingerprint каждой модели (`arch_fingerprint` + `training_hash`).

При повторном запуске sweep'а:
1. `Run.find_or_create()` проверяет Registry
2. `status=done` + `total_samples >= target_samples` → **пропускает**
3. Частично обучена → **возобновляет**
4. Не найдена → **создаёт новую**

Это позволяет:
- Перезапускать sweep без потери прогресса
- Менять JSON-конфиг — старые модели не переобучаются
- Разные sweep'ы переиспользуют одинаковые модели

**Важно:** дедупликация по `(architecture_fp, training_hash)` — UNIQUE constraint. Если меняешь `target_samples`, fingerprint тот же → не создаст новый run, а дотренирует существующий до нового target.

Подробно о Registry: [dev/REGISTRY.md](../dev/REGISTRY.md).

---

## Готовые конфиги (`configs/`)

```
rect_sweep_384m.json              # n=2..8, solve=b, budget=384M
pyramid_sweep_384m.json           # То же, pyramid
interleaved_sweep_384m.json       # То же, interleaved
trapezoid_sweep_384m.json         # То же, trapezoid
trapezoid_sweep_384m_n6.json      # trapezoid, фиксированный n=6

noise_sweep.json                  # noise_prob grid
noise_sweep_n6.json               # noise_prob grid, n=6 fixed
noise_sweep_compare_4M.json       # Сравнение уровней шума, 4M

bottleneck_sweep_4M.json          # bottleneck размеры
bs_sweep_4M_noise0025.json        # batch sizes

pre_norm_sweep_n468_4M.json       # pre-norm sweep
pre_norm_greedy_lr5e5.json        # pre-norm + greedy LR

asym_rect_*.json                  # Асимметричные архитектуры
enc_dec_ratio_sweep_n*.json       # Sweep соотношения enc/dec слоёв

vae_beta_sweep.json               # VAE β sweep
vae_bottleneck_sweep_n8.json      # VAE bottleneck sweep
vae_small_sweep.json              # VAE compact sweep
```

---

## Python-пресеты (`experiment/presets.py`)

Альтернатива JSON-конфигам — программная генерация:

```python
preset_ratio(budget_m)       # sweep n при заданном бюджете
preset_width(seq_len, n_hidden)  # sweep b при фиксированных n, seq_len
preset_batch(budget_m, n_hidden) # sweep batch_size
preset_binary()              # sweep seq_len
```

---

## Shorthand-аргументы sweep

| Флаг | Описание |
|------|----------|
| `--config` | JSON-конфиг (для `run`) |
| `--override` | `путь=значение`, можно несколько |
| `--vary` | `n=2,4,6` (grid) или имя параметра (binary) |
| `--range` | `lo hi` (binary) |
| `--solve` | `b` или `n` |
| `--fixed` | `n=7 b=4` — фиксация параметров |
| `--budget` | `40M` |
| `--seq-len` | 32 |
| `--lr` | 0.001 |
| `--scheduler` | onecycle |
| `--target-samples` | 5M |
| `--workspace` | sessions/sweep |
| `--device` | auto |
