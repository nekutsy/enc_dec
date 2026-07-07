# USAGE.md — Инструкция по использованию enc-dec

## Точка входа

```bash
bin/enc-dec <команда> [опции]
```

Доступные команды:

| Команда | Назначение |
|---------|-----------|
| `status` | Просмотр реестра экспериментов и запусков |
| `train` | Обучение одной модели |
| `sweep` | Grid/binary search по параметрам |
| `infer` | Интерактивная инференс-консоль |
| `overfit` | Overfit-тест на одном батче |
| `plot` | Построение графиков по логам |
| `resume` | Дослать все запуски до N семплов |

`bin/enc-dec <cmd> --help` покажет опции для конкретной команды.

Старые команды (`python cli/train.py`, `python cli/sweep.py` и т.д.) продолжают работать — просто `bin/enc-dec` короче и консистентнее.

---

## `status` — реестр

```bash
bin/enc-dec status                 # обзор: сколько экспериментов/запусков, последние done
bin/enc-dec status --runs          # все запуски с деталями
bin/enc-dec status --experiments   # список экспериментов
bin/enc-dec status --exp NAME      # детали конкретного эксперимента
bin/enc-dec status --run ID        # детали одного запуска (можно первые 6+ символов ID)
```

---

## `train` — одиночная модель

```bash
# Быстрый старт
bin/enc-dec train --n 3 --budget 160M --samples 50M

# Из JSON-конфига
bin/enc-dec train --config configs/n8_rect_bn160_50M.json

# Свой LR + plateau scheduler
bin/enc-dec train --n 2 --budget 160M --samples 120M --lr 0.002 --scheduler plateau

# С шумом
bin/enc-dec train --n 3 --budget 160M --samples 50M --noise-prob 0.25 --noise-std 3.0

# GPU force
bin/enc-dec train --n 3 --budget 160M --device cuda
```

**Авто-возобновление:** запусти ту же команду — подхватит чекпоинт и продолжит.

**Дедупликация:** если модель с такими же параметрами уже обучена до `target_samples` — пропустит (Registry find-or-create).

### Основные аргументы

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--n` | **обязателен**¹ | Количество скрытых слоёв с каждой стороны |
| `--budget` | **обязателен**¹ | `160M`, `40M` — целевое количество параметров |
| `--samples` | `50M` | Сколько семплов обучить |
| `--seq-len` | 128 | Длина окна в символах |
| `--bottleneck` | = seq_len | Размер бутылочного горла |
| `--b` | — | Ручное задание коэффициента ширины (вместо auto из бюджета) |
| `--shape` | rectangular | rectangular / pyramid / interleaved / trapezoid |
| `--lr` | 0.001 | Learning rate |
| `--scheduler` | onecycle | onecycle / plateau / cosine / greedy / greedy_simple / greedy_grad / none |
| `--optimizer` | adamw_fused | adamw_fused / adamw / lion / sophia / sgd / nag |
| `--noise-prob` | 0.0 | Доля зашумлённых символов |
| `--noise-std` | 3.0 | σ гауссовского шума на uint21 |
| `--activation` | silu | silu / relu / gelu / leaky_relu |
| `--normalization` | batchnorm | batchnorm / layernorm / none |
| `--device` | auto | auto / cuda / cpu |
| `--batch-size` | 256 | Размер батча |

¹ Не нужны при использовании `--config`.

---

## `sweep` — перебор параметров

См. [SWEEP.md](SWEEP.md) для деталей стратегий.

```bash
# Из JSON-конфига (основной способ)
bin/enc-dec sweep run --config configs/noise_sweep.json

# С оверрайдом
bin/enc-dec sweep run --config configs/rect_sweep_384m.json \
  --override model.seq_len=64 sweep.strategy=binary

# Shorthand grid
bin/enc-dec sweep grid --vary n=2,4,6,8,10 --solve b --budget 40M

# Shorthand binary
bin/enc-dec sweep binary --vary n --range 2 16 --solve b --budget 40M
```

---

## `infer` — инференс

```bash
bin/enc-dec infer            # CPU
bin/enc-dec infer --gpu      # GPU
```

Интерактивная консоль. Команды внутри:

| Команда | Описание |
|---------|----------|
| `<#>` | Загрузить модель по номеру из списка |
| `enc <text\|random\|@pos>` | Закодировать текст → латентный вектор |
| `dec` | Декодировать сохранённый латент → текст |
| `z` | Показать латентный вектор |
| `random` / `r` | Случайное окно → реконструкция |
| `full [pos]` | 20 окон подряд с позиции |
| `<любой текст>` | Прямая реконструкция |
| `q` | Выход |

---

## `overfit` — проверка архитектуры

Проверяет, может ли модель заучить один батч. Полезно для отладки архитектуры перед sweep.

```bash
# По умолчанию: seq=96, n=6, ~384M параметров
bin/enc-dec overfit

# Свои параметры
bin/enc-dec overfit --seq-len 128 --n 8 --budget 384M --max-steps 200000

# Pyramid shape
bin/enc-dec overfit --shape pyramid --seq-len 64 --n 4 --b 2.0 --lr 0.001

# Другой оптимизатор
bin/enc-dec overfit --optimizer adamw_fused --lr 0.001
```

Вывод:
- `✅ MODEL CAN OVERFIT` — архитектура рабочая
- `⚠ MODEL OVERFITS POORLY` — учится, но медленно
- `❌ MODEL CANNOT OVERFIT` — архитектура сломана

Логи пишутся в `sessions/_overfit/<model_name>/log.csv`.

---

## `plot` — графики

```bash
# Графики train_loss/val_loss/lr по всем запускам
bin/enc-dec plot runs

# Сравнение двух запусков (например, разные уровни шума)
bin/enc-dec plot noise 0.025 bbeda7548d05 0.25 c4cb0acad82f
```

Графики сохраняются в `sessions/plots/`.

---

## `resume` — дослать до N семплов

```bash
# Все запуски в sessions/runs/ → 12M семплов
bin/enc-dec resume

# Свой target
RESUME_TARGET=20000000 bin/enc-dec resume
```

Перезапускает `run.execute()` для каждой модели — подхватывает чекпоинты (веса, оптимизатор, шедулер).

---

## JSON-конфиги

Полный формат:

```json
{
  "name": "experiment_name",
  "model": {
    "seq_len": 128,
    "bottleneck": null,
    "activation": "silu",
    "normalization": "batchnorm",
    "shape": "rectangular",
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
    "warmup_fraction": 0.02,
    "optimizer": "adamw_fused",
    "weight_decay": 0.01,
    "early_stop_patience": 20,
    "train_ratio": 0.999,
    "num_workers": 2,
    "noise_prob": 0.0,
    "noise_std": 3.0
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
    "workspace": "sessions/sweep",
    "sweep_log": "sessions/global.csv",
    "device": "auto"
  }
}
```

---

## Структура sessions/

```
sessions/
├── runs/
│   ├── cce656d8f25a → cce656d8f25a-rect_s128_n8_b2.0298  (symlink, backward compat)
│   ├── cce656d8f25a-rect_s128_n8_b2.0298/
│   │   ├── model.pth / best.pth       # чекпоинты
│   │   ├── model.opt / model.sch      # оптимизатор / шедулер
│   │   ├── meta.json / result.json    # конфиг / финальные метрики
│   │   ├── log.csv / train.log        # метрики по шагам / текстовый лог
│   │   └── ...
│   └── ...
├── experiments/
│   ├── noise_sweep_n6/
│   │   └── config.json                # копия SweepConfig на момент запуска
│   └── bs_sweep_4M_noise0025/
│       └── config.json
├── registry.db                        # SQLite (в .gitignore)
├── plots/                             # output plot-скриптов
├── _overfit/                          # overfit-тесты
└── _ad_hoc_artifacts/                 # легаси/ручные запуски
```

Шаблон имени run-директории: `{12-char_hash}-{model_name}`.
model_name генерится из архитектуры, например `rect_s128_n6_b2.2013`.

---

## Общий workflow

1. **Проверить архитектуру:** `enc-dec overfit --seq-len 128 --n 8 --budget 384M`
2. **Запустить sweep из конфига:** `enc-dec sweep run --config configs/noise_sweep.json`
3. **Следить за прогрессом:** `enc-dec status`
4. **Если нужно дослать:** `enc-dec resume`
5. **Посмотреть графики:** `enc-dec plot runs`
6. **Протестировать модель:** `enc-dec infer`
