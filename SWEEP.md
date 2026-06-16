# SWEEP.md — Документация по sweep-системе

## Быстрый старт

```bash
# Из готового конфига
python sweep.py run --config configs/ratio_40m.json

# Shorthand (без файла)
python sweep.py grid --vary n=2,4,6,8,10 --solve b --budget 40M

# Конфиг + оверрайд
python sweep.py run --config configs/ratio_40m.json \
  --override model.seq_len=64 sweep.strategy=binary sweep.values=1,10
```

---

## Интерфейсы

### `python sweep.py run` — из конфиг-файла

```bash
python sweep.py run --config <путь> [--override путь=значение ...]
```

**Аргументы:**

| Флаг | Описание | Пример |
|------|----------|--------|
| `--config` | Путь к JSON-конфигу | `configs/ratio_40m.json` |
| `--override` | Точечный оверрайд (можно несколько) | `model.seq_len=64 sweep.solve=n` |

Оверрайды используют dotted-нотацию: `model.seq_len`, `training.lr`, `sweep.solve`, `output.workspace`. Числовые значения автоматически кастуются к нужному типу.

### `python sweep.py grid` — shorthand grid

```bash
python sweep.py grid --vary <параметр>=<значения> [опции]
```

**Аргументы:**

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--vary` | **обязателен** | Какой параметр варьировать, и какие значения<br>`n=2,3,4,5,6` или `b=0.14,0.33,1,2,4,8` |
| `--solve` | нет | Derived-параметр: `b` (вычислить b из бюджета) или `n` |
| `--fixed` | нет | Фиксированные параметры: `n=7 b=4 seq_len=32 lr=0.001` |
| `--budget` | нет | Бюджет параметров: `40M`, `80M` |
| `--seq-len` | 32 | seq_len модели |
| `--bottleneck` | равен seq_len | Размер bottleneck |
| `--lr` | 0.001 | Learning rate |
| `--scheduler` | cosine | LR-шедулер: `cosine`, `plateau`, `none` |
| `--target-symbols` | 120M | Сколько символов обучать |
| `--workspace` | sessions/sweep | Папка для чекпоинтов и CSV |
| `--sweep-log` | sessions/sweep_summary.csv | Путь к сводному CSV |
| `--device` | auto | `auto`, `cuda`, `cpu` |
| `--batch-size` | adaptive | Фиксированный batch size (или auto) |
| `--binary-on` | нет | Для grid: параметр, внутри которого бинарный поиск |
| `--range` | нет | Границы бинарного поиска: `1 16` |

### `python sweep.py binary` — shorthand binary

```bash
python sweep.py binary --vary <параметр> --range <min> <max> [опции]
```

Те же опции, что у grid, но `--vary` — одно имя (без `=` и значений), диапазон — через `--range`.

---

## Стратегии

### Grid

Перебирает все значения параметра, для каждого обучает модель.

```
python sweep.py grid --vary n=2,3,4,5,6,7 --solve b --budget 40M
→ бьёт 6 моделей: n=2..7, для каждого b подбирается под 40M параметров
```

### Binary

Бинарный поиск оптимума. Пробует границы, затем делит пополам между лучшим и вторым.
Останавливается, когда лучший и второй — соседи.

```
python sweep.py binary --vary n --range 1 16 --fixed b=4
→ бинарный поиск n ∈ [1, 16] при фиксированном b=4
```

### Grid × Binary (embedded)

Внешний grid, внутри каждой ячейки — бинарный поиск по другому параметру.

```bash
python sweep.py grid --vary seq_len=4,8,16,32,64,128 \
  --binary-on n --range 1 16 --fixed b=4
```

Для каждого `seq_len` запускается бинарный поиск оптимального `n`.

---

## Конфиг-файл (JSON)

### Полная схема

```json
{
  "name": "experiment_name",
  
  "model": {
    "seq_len": 32,
    "bottleneck": null,
    "activation": "silu",
    "normalization": "batchnorm",
    "init": "orthogonal",
    "init_gain": 0.5,
    "dropout": 0.0
  },
  
  "training": {
    "target_symbols": 120000000,
    "lr": 0.001,
    "grad_clip": 1.0,
    "scheduler": "cosine",
    "warmup_fraction": 0.05,
    "optimizer": "adamw_fused",
    "weight_decay": 0.01,
    "early_stop_patience": 3,
    "train_ratio": 0.99
  },
  
  "sweep": {
    "strategy": "grid",
    "vary": "n",
    "values": [2, 4, 6, 8, 10],
    "solve": "b",
    "budget": 40000000,
    "fixed": {}
  },
  
  "output": {
    "workspace": "sessions/ratio40",
    "sweep_log": "sessions/ratio40_sweep_summary.csv",
    "device": "auto",
    "batch_size": null
  }
}
```

### Поля `model`

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `seq_len` | int | 32 | Длина окна в символах; input_dim = seq_len × 21 |
| `bottleneck` | int\|null | `null` | Размер bottleneck; `null` → равен seq_len |
| `activation` | str | `silu` | `silu` \| `relu` \| `gelu` \| `leaky_relu` |
| `normalization` | str | `batchnorm` | `batchnorm` \| `layernorm` \| `none` |
| `init` | str | `orthogonal` | Метод инициализации весов |
| `init_gain` | float | 0.5 | Gain для orthogonal init |
| `dropout` | float | 0.0 | Dropout probability |

### Поля `training`

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `target_symbols` | int | 120M | Сколько total символов обучать (≠ эпох) |
| `lr` | float | 0.001 | Learning rate |
| `grad_clip` | float | 1.0 | Max gradient norm |
| `scheduler` | str | `cosine` | `cosine` \| `plateau` \| `none` |
| `warmup_fraction` | float | 0.05 | Доля шагов на warmup |
| `optimizer` | str | `adamw_fused` | `adamw_fused` \| `adamw` \| `sgd` |
| `weight_decay` | float | 0.01 | Weight decay |
| `early_stop_patience` | int | 3 | Чекпоинтов без улучшения до остановки |
| `train_ratio` | float | 0.99 | Доля данных в train |

### Поля `sweep`

| Поле | Тип | Описание |
|------|-----|----------|
| `strategy` | str | `grid` \| `binary` |
| `vary` | str | Имя варьируемого параметра |
| `values` | list | Grid: список значений; binary: `[min, max]` |
| `solve` | str\|null | `b` (derive b from budget) \| `n` \| `null` |
| `budget` | int\|null | Целевое кол-во параметров |
| `fixed` | dict | Дополнительные фиксированные параметры, например `{"n": 7}` |

### Поля `output`

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `workspace` | str | sessions/sweep | Куда писать чекпоинты |
| `sweep_log` | str | sessions/sweep_summary.csv | Сводный CSV |
| `device` | str | `auto` | `auto` \| `cuda` \| `cpu` |
| `batch_size` | int\|null | `null` | `null` → adaptive; иначе фиксированный |

---

## Примеры использования

### 1. Ratio sweep: сетка по n, b из бюджета

```bash
# 40M бюджет
python sweep.py run --config configs/ratio_40m.json

# Тоже самое shorthand
python sweep.py grid --vary n=2,4,6,8,10 --solve b --budget 40M
```

### 2. Width sweep: сетка по b, фиксированный n

```bash
python sweep.py run --config configs/width_sweep.json

# Или shorthand
python sweep.py grid --vary b=0.14,0.33,1,2,4,8 --fixed n=7
```

### 3. Batch-size sweep

```bash
python sweep.py run --config configs/batch_sweep.json

# Shorthand
python sweep.py grid --vary batch_size=64,128,256,512,1024,2048,4096,8192 \
  --solve b --fixed n=3 --budget 20M
```

### 4. Binary search по n при фиксированном b

```bash
python sweep.py binary --vary n --range 1 16 --fixed b=4 --budget 40M
```

### 5. Grid по seq_len с embedded binary по n

```bash
python sweep.py grid --vary seq_len=4,8,16,32,64,128 \
  --binary-on n --range 1 16 --fixed b=4
```

### 6. Быстрый эксперимент с оверрайдом

```bash
# Взять базовый конфиг ratio_40m, но:
#  - seq_len=64 вместо 32
#  - bottleneck независимый (16 вместо seq_len)
#  - gelu вместо silu
python sweep.py run --config configs/ratio_40m.json \
  --override model.seq_len=64 model.bottleneck=16 model.activation=gelu
```

### 7. Свой конфиг с нуля

Скопируй `configs/ratio_40m.json`, отредактируй — и `python sweep.py run --config my_new.json`.

---

## Как это работает

### Поток данных

```
JSON config / CLI args
        │
        ▼
   SweepConfig (dataclass)
        │
        ▼
   SweepRunner.run()
        │
        ├─ grid:  for each value → resolve_architecture() → train_one()
        │
        └─ binary:  boundary probes → binary search → resolve → train_one
                              │
                              ▼
                    resolve_architecture()
                      solve_b_for_n()  или  solve_n_for_b()
                              │
                              ▼
                      make_rectangular()
                        [input_dim] + [hidden]×n + [bottleneck] + [hidden]×n + [input_dim]
                              │
                              ▼
                      train_one()
                        Autoencoder(sizes) → compile → AdamW → обучение
                              │
                              ▼
                      CSV log + чекпоинты
```

### Resolve architecture

В зависимости от того, что задано:

| Дано | Результат |
|------|-----------|
| `solve="b"`, `n`, `budget` | b вычисляется бинарным поиском под бюджет |
| `solve="n"`, `b`, `budget` | n вычисляется бинарным поиском под бюджет |
| `n`, `b` | hidden_dim = int(input_dim × b), без поиска |
| `b` + `budget` (без n) | n вычисляется, остальное фиксировано |

### Resume

Модели автоматически возобновляются с последнего чекпоинта, если:

1. CSV-файл существует
2. `total_symbols` в CSV < `target_symbols`

Модели, уже достигшие `target_symbols`, пропускаются.

### Формат CSV-лога

Единый для всех sweep'ов:

```
sweep_type,vary_param,vary_value,seq_len,n_hidden,b,hidden_dim,bottleneck,
params,batch_size,total_symbols,final_train_loss,final_val_loss,status,duration_seconds
```

---

## GPU safety

- Перед стартом: `gpu_health_check()` — проверяет доступность CUDA
- После каждой модели: `torch.cuda.synchronize()` + `empty_cache()`
- OOM обрабатывается: модель пропускается, GPU чистится
- Ctrl+C: graceful cleanup + сохранение чекпоинта
- `atexit` handler — страховка от падений без cleanup

---

## Готовые пресеты

| Файл | Что делает |
|------|-----------|
| `configs/ratio_20m.json` | Grid n=2..6, solve b, бюджет 20M |
| `configs/ratio_40m.json` | Grid n=2,4,6,8,10, solve b, бюджет 40M |
| `configs/ratio_80m.json` | Grid n=2..16, solve b, бюджет 80M |
| `configs/binary_search.json` | Grid seq_len=4..128, для каждого — binary по n |
| `configs/width_sweep.json` | Grid b={1/7,1/3,1,2,4,8}, фикс n=7 |
| `configs/batch_sweep.json` | Grid batch_size=64..8192, solve b, бюджет 20M |
