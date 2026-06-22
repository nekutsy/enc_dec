# SWEEP.md — Свипы и тренировка

## `train.py` — одиночная модель

Универсальный CLI для обучения одной модели с авто-возобновлением. Заменил `train_best.py` + `resume_n4.py`.

```bash
# Базовый запуск
python train.py --n 3 --budget 160M --samples 50M

# Plateau + свой LR
python train.py --n 2 --budget 160M --samples 120M \
  --lr 0.002 --scheduler plateau --early-stop 10

# Возобновление (автоматическое — просто запусти ещё раз)
python train.py --n 3 --budget 160M --samples 50M

# Сброс LR при возобновлении (веса остаются, оптимизатор/шедулер — с нуля)
python train.py --n 4 --budget 160M --samples 100M --reset-lr --lr 0.001

# Полный fresh start (игнорировать чекпоинты)
python train.py --n 3 --budget 160M --samples 50M --fresh

# Из JSON-конфига
python train.py --config train_best.json

# Кастомная папка
python train.py --n 3 --budget 160M --samples 50M --workspace sessions/my_exp
```

### Все аргументы

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--config` | — | JSON-конфиг (CLI-флаги переопределяют) |
| `--fresh` | false | Игнорировать чекпоинты, начать с нуля |
| `--reset-lr` | false | Сбросить оптимизатор/шедулер, сохранить веса |

**Архитектура:**

| Флаг | По умолчанию |
|------|-------------|
| `--seq-len` | 128 |
| `--n` | **обязателен** |
| `--b` | — (auto из бюджета) |
| `--budget` | **обязателен**, e.g. `160M` |
| `--bottleneck` | равен seq_len |
| `--activation` | silu |
| `--normalization` | batchnorm |
| `--init-gain` | 1.0 |
| `--dropout` | 0.0 |
| `--norm-bottleneck` | false |
| `--norm-last` | false |

**Обучение:**

| Флаг | По умолчанию |
|------|-------------|
| `--samples` | 50M |
| `--batch-size` | 256 |
| `--lr` | 0.001 |
| `--scheduler` | onecycle |
| `--optimizer` | adamw_fused |
| `--weight-decay` | 0.01 |
| `--grad-clip` | 1.0 |
| `--early-stop` | 3 |
| `--num-workers` | 2 |
| `--train-ratio` | 0.999 |
| `--no-val` | false (skip val in CSV) |

**Вывод:**

| Флаг | По умолчанию |
|------|-------------|
| `--workspace` | sessions/train |
| `--name` | auto (`n{val}_s{seq_len}`) |
| `--device` | auto |

---

## `sweep.py` — сетка/бинарный поиск

### Быстрый старт

```bash
# Из готового конфига
python sweep.py run --config configs/ratio_40m.json

# Shorthand (без файла)
python sweep.py grid --vary n=2,4,6,8,10 --solve b --budget 40M

# Конфиг + оверрайд
python sweep.py run --config configs/ratio_40m.json \
  --override model.seq_len=64 sweep.strategy=binary sweep.values=1,10
```

### `python sweep.py run` — из JSON-конфига

```bash
python sweep.py run --config <путь> [--override путь=значение ...]
```

Оверрайды: dotted-нотация (`model.seq_len=64`, `training.lr=0.002`, `sweep.solve=n`).

### `python sweep.py grid` — shorthand grid

```bash
python sweep.py grid --vary n=2,3,4,5,6 [опции]
```

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--vary` | **обязателен** | `n=2,3,4,5` или `b=0.14,0.33,1,2,4,8` |
| `--solve` | нет | `b` (вычислить b из бюджета) или `n` |
| `--fixed` | нет | `n=7 b=4 lr=0.001` |
| `--budget` | нет | `40M`, `160M` |
| `--seq-len` | 32 | |
| `--bottleneck` | равен seq_len | |
| `--lr` | 0.001 | |
| `--scheduler` | onecycle | onecycle / plateau / cosine / none |
| `--target-samples` | 5M | `120M` |
| `--workspace` | sessions/sweep | |
| `--sweep-log` | sessions/sweep_summary.csv | |
| `--batch-size` | 256 | |
| `--binary-on` | нет | Для grid: вложенный бинарный поиск |
| `--range` | нет | Границы бинарного поиска |

### `python sweep.py binary` — shorthand binary

```bash
python sweep.py binary --vary n --range 1 16 [опции]
```

---

## JSON-конфиг

```json
{
  "name": "experiment_name",
  "model": {
    "seq_len": 32,
    "bottleneck": null,
    "activation": "silu",
    "normalization": "batchnorm",
    "init_gain": 1.0,
    "dropout": 0.0,
    "norm_bottleneck": false,
    "norm_last": false
  },
  "training": {
    "target_samples": 5000000,
    "batch_size": 256,
    "lr": 0.001,
    "grad_clip": 1.0,
    "scheduler": "onecycle",
    "warmup_fraction": 0.05,
    "optimizer": "adamw_fused",
    "weight_decay": 0.01,
    "decay_linear_only": true,
    "early_stop_patience": 3,
    "train_ratio": 0.999,
    "num_workers": 2
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
    "sweep_log": "sessions/ratio40_summary.csv",
    "device": "auto"
  }
}
```

---

## Ключевые изменения за июнь 2026

### Plateau scheduler
- factor 0.7 (вместо 0.5), min_lr=1e-6
- Состояние сохраняется в `.sch` и восстанавливается при рестарте
- Всегда смотрит на val loss (даже при `no_val=True` в sweep)

### Валидация
- `train_ratio` по умолчанию 0.999 (0.1% валидационной выборки)
- `no_val` только пропускает запись val в CSV (scheduler и early-stop всё равно получают val loss)
- `_best.pth` сохраняется всегда (раньше — только при `no_val=False`)
- Early-stopping работает всегда

### Файлы
- `train.py` — заменил `train_best.py` + `resume_n4.py`
- `sweep_gain.py`, `sweep_norm.py`, `sweep_norm_none.py` — удалены (всё через `train.py` + оверрайды или JSON-конфиг)
