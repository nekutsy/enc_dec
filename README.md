# enc_dec — текст в латентный вектор через автоэнкодер

Симметричный автоэнкодер, сжимающий текст (набор символов Unicode) в компактный
латентный вектор и восстанавливающий его обратно. Обучается как бинарный
классификатор на 21 бите каждого Unicode codepoint.

**Сжатие:** 21:1 (128 символов × 21 бит → 128-мерный латентный вектор).

---

## Архитектура

### Autoencoder (`model.py`)

Симметричный полносвязный автоэнкодер с настраиваемыми:
- **Глубина** — задаётся массивом `layer_sizes`, середина = bottleneck
- **Форма** — rectangular (все скрытые слои одной ширины) или pyramid (сужающаяся к центру)
- **Активация** — `silu` (по умолчанию), `relu`, `gelu`, `leaky_relu`
- **Нормализация** — `batchnorm` (BatchNorm1d), `layernorm`, либо `none`
- **Инициализация** — orthogonal, настраиваемый gain
- **Dropout** — опциональный

Каждый скрытый слой: `Linear → Norm → Activation → Dropout(опционально)`.

Выход декодера — сырые логиты (без sigmoid), подаются в `BCEWithLogitsLoss`.

```
Input (D) → ... → Bottleneck (B) → ... → Output (D)
       └─ encoder ─┘          └─ decoder ─┘
```

#### Две формы архитектуры

| Форма | Описание | Функция |
|-------|----------|---------|
| **Rectangular** | `[D] + [H]×n + [B] + [H]×n + [D]` — скрытые слои одинаковой ширины H | `make_rectangular()` |
| **Pyramid** | `[D] → h₁ → … → hₙ → [B] → hₙ → … → h₁ → [D]` — сужающаяся к центру | `make_pyramid()` |

Где:
- `D` = `seq_len × 21` (input_dim), например 128×21 = 2688
- `B` = `bottleneck` (по умолчанию = seq_len)
- `H` = `hidden_dim` ≈ `D × b` (b — коэффициент ширины)
- `n` = количество скрытых слоёв с каждой стороны

#### Создание модели

```python
from model import Autoencoder

# 11-layer autoencoder: 2688 → ... → 128 → ... → 2688
sizes = [2688, 5376, 10752, 5376, 5376, 128, 5376, 5376, 10752, 5376, 2688]
model = Autoencoder(sizes, activation='silu', normalization='batchnorm', dropout=0.1)

# Энкодинг
z = model.encode(x)  # z.shape = (batch, 128)

# Декодинг (возвращает логиты — нужен sigmoid для вероятностей)
logits = model.decode(z)

# Сквозной проход
reconstructed = model(x)
```

---

## Data Pipeline (`data.py`)

### Кодировка Unicode-21

Каждый символ ∈ [U+0000, U+1FFFFF] → 21 бит (старший бит первый).

### SlidingWindowDataset

Окна со **stride=1 символ** — каждый символ появляется в `seq_len` окнах.
Реализован через `torch.as_strided` на float32-тензоре, без копирования.

### Двухуровневое кэширование

| Уровень | Хранение | Размер (~50M символов) |
|---------|----------|------------------------|
| Диск | uint8 packed (`data/cache/full_bits.u8`) | ~0.13 GB |
| RAM | float32 unpacked | ~4.2 GB |

При первом запуске: биты упаковываются в uint8 на диск → при загрузке распаковываются в float32.
Автоматическая миграция со старого float32-кэша.

---

## Файлы

```
enc_dec/
├── model.py              # Autoencoder (nn.Module)
├── configs.py            # UNICODE_BITS = 21
├── sweep_config.py       # ModelConfig, TrainConfig, SweepConfig, OutputConfig (датаклассы)
├── data.py               # Загрузка текста, Unicode-21, SlidingWindowDataset, кэш
├── train.py              # CLI тренировки одной модели (основной интерфейс)
├── sweep.py              # CLI sweep-раннер: grid, binary, grid×binary
├── sweep_lib.py          # resolve_architecture, train_one, build_optimizer, Lion, Sophia
├── infer_test.py         # Интерактивная инференс-консоль
├── logger.py             # TrainingLogger, GlobalLogger, LoggerConfig
├── utils.py              # CUDA-безопасная очистка, проверка GPU
├── SWEEP.md              # Документация по sweep-системе
├── README.md             # ← этот файл
├── training/             # Пакет тренировочного цикла
│   ├── loop.py           # Основной цикл: валидация, чекпоинты, early-stopping
│   ├── step.py           # Один шаг: forward + backward + AMP + grad clip
│   ├── scheduler.py      # LR-шедулеры: onecycle, plateau, cosine, greedy, none
│   └── checkpoint.py     # Сохранение/загрузка чекпоинтов и состояния early-stopping
├── sessions/             # Чекпоинты (.pth, .opt, .sch) и CSV-логи
├── data/
│   ├── dataset/          # .txt файлы (~50M символов русской прозы)
│   └── cache/            # full_bits.u8 (автосоздаётся)
└── archive/              # История исследований
```

---

## Быстрый старт

### Тренировка одной модели

```bash
# Базовый запуск
python train.py --n 3 --budget 160M --samples 50M

# Plateau-шедулер + свой LR
python train.py --n 2 --budget 160M --samples 120M --lr 0.002 --scheduler plateau

# Авто-возобновление (просто запусти ещё раз)
python train.py --n 3 --budget 160M --samples 50M

# Fresh start (игнорировать чекпоинты)
python train.py --n 3 --budget 160M --samples 50M --fresh

# Сброс LR при возобновлении
python train.py --n 4 --budget 160M --samples 100M --reset-lr --lr 0.001
```

Основные параметры по умолчанию:
- `seq_len=128`, `bottleneck=seq_len`, `activation=silu`, `normalization=batchnorm`
- `lr=0.001`, `batch_size=256`, `optimizer=adamw_fused`
- `scheduler=onecycle`, `init_gain=1.0`

### Sweep нескольких моделей

См. [`SWEEP.md`](SWEEP.md) — полная документация.

```bash
# Grid: перебрать n=2,4,6,8,10, b подобрать под 40M параметров
python sweep.py grid --vary n=2,4,6,8,10 --solve b --budget 40M

# Из JSON-конфига
python sweep.py run --config train_n8_384m_v1.json

# Конфиг + оверрайд
python sweep.py run --config train_n8_384m_v1.json --override model.seq_len=64
```

### Инференс: протестировать модели

```bash
python infer_test.py
# или на GPU:
python infer_test.py --gpu
```

**Команды `infer_test.py`:**

| Команда | Описание |
|---------|----------|
| `<#>` / `load <#>` | Загрузить модель по номеру |
| `val <#> <#> ...` | Быстрая валидация на 1% датасета |
| `enc <text\|random\|@pos>` | Закодировать текст → латентный вектор |
| `dec` / `dec <values>` | Декодировать латент → текст |
| `z` / `latent` | Показать сохранённый латентный вектор |
| `random` / `r` | Случайное окно → реконструкция |
| `full [pos]` | 20 окон подряд с позиции |
| `<любой текст>` | Прямая реконструкция |
| `q` / `quit` | Выход |

---

## Детали тренировки

### Функция потерь

`BCEWithLogitsLoss` — каждый из 21 бита классифицируется независимо.
Декодер выдаёт логиты, лосс применяет sigmoid внутри.

### LR-шедулеры

| Шедулер | Описание |
|---------|----------|
| `onecycle` | OneCycleLR — быстрый разгон + плавное затухание |
| `plateau` | ReduceLROnPlateau — снижение LR при выходе на плато |
| `cosine` | CosineAnnealingLR с warmup |
| `greedy` | Zeroth-order GreedyLR — адаптивный с пробами (bidirectional) |
| `none` | Константный LR |

При рестарте warmup пропускается — LR уже на рабочем уровне.

### Оптимизаторы

| Оптимизатор | Примечание |
|-------------|------------|
| `adamw_fused` | По умолчанию; fused-реализация на CUDA |
| `adamw` | Стандартный AdamW |
| `lion` | EvoLved Sign Momentum (Google) |
| `sophia` | Second-order clipped optimizer |
| `sgd` | SGD + momentum |
| `nag` | SGD + Nesterov momentum |

### Early stopping

По валидационной ошибке. Если `patience` чекпоинтов без улучшения — остановка.
Лучшая модель сохраняется как `*_best.pth`. Состояние восстанавливается при рестарте.

### Mixed precision (AMP)

Автоматически на CUDA: `autocast(bfloat16)` + `GradScaler`.
Даталоадер: `pin_memory=True` + `non_blocking=True`.

### CUDA-безопасность

- Signal handlers только выставляют флаг, не вызывают CUDA
- При выходе: `torch.cuda.synchronize()` + `empty_cache()`
- `atexit` handler как страховка при падении
- OOM protection: GPU чистится, модель пропускается в sweep

---

## Конфигурация

Детальная структура конфигов — в [`sweep_config.py`](sweep_config.py):

- `ModelConfig` — архитектура, активации, нормализация
- `TrainConfig` — оптимизатор, LR, шедулер, batch, early-stopping
- `SweepSpec` — стратегия (grid/binary), варьируемый параметр, бюджет
- `OutputConfig` — пути, устройство
- `SweepConfig` — объединяет всё; сериализуется в JSON

---

## Результаты исследований

См. [`archive/RESEARCH_2026-06-18.md`](archive/RESEARCH_2026-06-18.md) — актуальные результаты sweep-экспериментов.
