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
| `lr-find` | LR range test для одной архитектуры |

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

# С шумом (фиксированный 25%)
bin/enc-dec train --n 3 --budget 160M --samples 50M --noise-prob 0.25

# С шумом — per-batch диапазон (каждый сэмпл получает свой уровень шума)
bin/enc-dec train --n 3 --budget 160M --samples 50M \
  --noise-prob-min 0.0 --noise-prob-max 0.3

# Указать стратегию сэмплинга и stride
bin/enc-dec train --n 3 --budget 160M --samples 50M \
  --noise-strategy uniform --noise-stride 128

# GPU force
bin/enc-dec train --n 3 --budget 160M --device cuda
```

**Авто-возобновление:** запусти ту же команду — подхватит чекпоинт и продолжит.

**Fine-tune с претрейном:** `--pretrain-from <run_id>` — начать обучение с весов существующего рана.
Архитектура автоматически наследуется от донора (из `meta.json`). Оптимизатор и шедулер — с нуля.

```bash
enc-dec train --pretrain-from cce656d8f25a --samples 50M --lr 0.0001
enc-dec train --pretrain-from cce656d8f25a --samples 100M --lr 0.0001 --scheduler greedy --noise-prob 0.25
```

При указании `--pretrain-from` игнорируются: `--n`, `--b`, `--seq-len`, `--shape`, `--activation`,
`--normalization` — они берутся из донора. Остальные аргументы (`--samples`, `--lr`, `--noise-prob`,
`--scheduler`, ...) применяются как обычно.

Донор не модифицируется — это новый run со своим `run_id`, директорией и логом.

**Дедупликация:** если модель с такими же параметрами уже обучена до `target_samples` — пропустит (Registry find-or-create).

### Основные аргументы

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--n` | **обязателен**¹ | Количество скрытых слоёв с каждой стороны |
| `--enc-n` | = n | Слоёв в энкодере (переопределяет `--n` для encoder) |
| `--dec-n` | = enc_n | Слоёв в декодере (по умолчанию равно encoder) |
| `--budget` | **обязателен**¹ | `160M`, `40M` — целевое количество параметров |
| `--samples` | `50M` | Сколько семплов обучить |
| `--seq-len` | 128 | Длина окна в символах |
| `--bottleneck` | = seq_len | Размер бутылочного горла |
| `--b` | — | Ручное задание коэффициента ширины (вместо auto из бюджета) |
| `--shape` | rectangular | rectangular / pyramid / interleaved / trapezoid |
| `--lr` | 0.001 | Learning rate |
| `--scheduler` | onecycle | onecycle / plateau / cosine / greedy / greedy_simple / greedy_grad / none |
| `--optimizer` | adamw_fused | adamw_fused / adamw / lion / sophia / sgd / nag |
| `--noise-prob` | — | Фиксированная доля зашумлённых символов (0..1, синоним min=max) |
| `--noise-prob-min` | 0.0 | Нижняя граница per-batch сэмплинга prob |
| `--noise-prob-max` | 0.0 | Верхняя граница (0→disabled) |
| `--noise-std` | — | Фиксированное σ шума (синоним min=max) |
| `--noise-std-min` | 3.0 | Нижняя граница per-batch сэмплинга std |
| `--noise-std-max` | 3.0 | Верхняя граница |
| `--noise-strategy` | linear | linear (интерполяция по idx % stride) / uniform (случайный) |
| `--noise-stride` | 256 | Шаг интерполяции для linear (0→=batch_size) |
| `--vae` | false | Включить VAE-режим (μ/logvar head + KL loss) |
| `--vae-beta` | 1.0 | β вес для KL-терма (0.5 = слабее регуляризация, 2.0 = сильнее) |
| `--activation` | silu | silu / relu / gelu / leaky_relu |
| `--normalization` | batchnorm | batchnorm / layernorm / none |
| `--device` | auto | auto / cuda / cpu |
| `--batch-size` | 256 | Размер батча |
| `--pretrain-from` | — | Run ID донора для fine-tune с его весов |

¹ Не нужны при использовании `--config`.

### Асимметричные архитектуры

Можно задавать разное количество слоёв в энкодере и декодере:

```bash
# Энкодер 8 слоёв, декодер 3 слоя (тот же бюджет параметров)
bin/enc-dec train --enc-n 8 --dec-n 3 --budget 40M --samples 5M

# Sweep по разным комбинациям
bin/enc-dec sweep grid --vary enc_n=3,5,7 --fixed dec_n=2 --solve b --budget 40M

# Overfit с асимметрией
bin/enc-dec overfit --seq-len 32 --enc-n 4 --dec-n 1 --b 1.5

# LR finder
bin/enc-dec lr-find --seq-len 64 --enc-n 6 --dec-n 2 --budget 40M --shape rectangular
```

`--n` — shorthand, задаёт одинаковое количество слоёв с обеих сторон.
`--enc-n` / `--dec-n` — точный контроль.

### VAE-режим

Опциональный вариационный режим. Энкодер выдаёт μ и log σ², декодер обучается на семплах из N(μ, σ²).

```bash
# VAE с β=1 (стандартный)
bin/enc-dec train --n 6 --budget 384M --samples 50M --vae --vae-beta 1.0

# β-VAE с ослабленной регуляризацией (меньше KL → лучше реконструкция)
bin/enc-dec train --n 6 --budget 160M --samples 50M --vae --vae-beta 0.1

# Sweep по разным β
bin/enc-dec sweep grid --vary noise_prob=0.0,0.25 --fixed n=6 --solve b --budget 40M \
  --override model.vae=true training.vae_beta=0.5
```

**Отличия от обычного автоэнкодера:**
- `forward()` возвращает `(reconstruction, μ, log σ²)` — тройку вместо одного тензора
- `encode()` возвращает μ (детерминированно)
- Доступен `model.sample(N)` — генерация из N(0,I) приора
- Loss: `BCE(recon, target) + β · KL(μ, σ² || N(0,I))`
- Архитектура: два дополнительных линейных слоя на bottleneck для μ и log σ²
- Число параметров: bottleneck не меняется (μ и log σ² той же размерности) — overhead минимален

**β-балансировка:**
- `β = 0` — обычный автоэнкодер (KL отключён, VAE вырождается)
- `β < 1` — ослабленная регуляризация, лучше реконструкция, хуже генерация
- `β = 1` — стандартный VAE
- `β > 1` — усиленная регуляризация, лучше диспентанглинг латентного пространства

**Инференс VAE-модели:**
```
> 0                    # загрузить VAE-модель
> sample               # сгенерировать 1 сэмпл из приора
> sample 5             # сгенерировать 5 сэмплов
> info                 # покажет mode: VAE  β=1.0
```

### Зашумление (denoising)

Модель обучается как denoising autoencoder: вход зашумляется, target остаётся чистым.
Шум добавляется на уровне uint21-значений: с вероятностью `noise_prob` к целочисленному
значению символа добавляется `N(0, noise_std²)`, затем округление и clamp в `[0, 2²¹−1]`.

**Per-batch сэмплинг:** `noise_prob` и `noise_std` не фиксированы, а берутся из диапазона
`[min, max]` для каждого семпла. Две стратегии:

| Стратегия | Поведение |
|-----------|----------|
| `linear` (default) | `noise = min + (max - min) · (idx % stride) / (stride - 1)`. Каждый батч гарантированно покрывает весь диапазон. Детерминированно по индексу семпла. |
| `uniform` | Случайный `uniform(min, max)` для каждого семпла. |

**Семплы с `noise_prob ≤ 0` сразу возвращаются без шума** — при `min=0` часть батча всегда чистая.

```bash
# Фиксированный шум (min=max) — старый синтаксис
bin/enc-dec train --noise-prob 0.25

# Диапазон [0, 0.5] с линейной интерполяцией (default)
bin/enc-dec train --noise-prob-min 0.0 --noise-prob-max 0.5

# То же, но uniform random
bin/enc-dec train --noise-prob-min 0.0 --noise-prob-max 0.5 --noise-strategy uniform

# Варьировать и prob, и std одновременно
bin/enc-dec train --noise-prob-min 0.0 --noise-prob-max 0.3 \
  --noise-std-min 1.0 --noise-std-max 5.0
```

**Sweep с разными режимами шума:**

```json
{
  "sweep": {
    "vary": "noise_prob_min",
    "values": [0.0, 0.25, 0.5, [0.0, 0.5]]
  }
}
```

Скалярные значения (0.0, 0.25, 0.5) задают фиксированный min=max.
Список `[0.0, 0.5]` задаёт диапазон — min≠max, linear стратегия по умолчанию.

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
| `sample [N]` | Генерация N сэмплов из приора (VAE only) |
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
    "norm_last": false,
    "vae": false
  },
  "training": {
    "pretrain_run_id": "",
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
    "noise_prob_min": 0.0,
    "noise_prob_max": 0.0,
    "noise_std_min": 3.0,
    "noise_std_max": 3.0,
    "noise_strategy": "linear",
    "noise_stride": 256,
    "vae_beta": 1.0
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
