# enc_dec — текст в латентный вектор через автоэнкодер

Симметричный автоэнкодер, сжимающий текст (набор символов Unicode) в компактный
латентный вектор и восстанавливающий его обратно. Обучается как бинарный
классификатор на 21 бите каждого Unicode codepoint.

**Сжатие:** 21:1 (128 символов × 21 бит → 128-мерный латентный вектор).

---

## Архитектура

### Autoencoder (`model.py`)

Симметричный полносвязный автоэнкодер с настраиваемыми:
- **Глубина** — задаётся массивом `layer_sizes`, середина массива = bottleneck
- **Активация** — `silu` (по умолчанию), `relu`, `gelu`, `leaky_relu`
- **Нормализация** — `batchnorm` (BatchNorm1d), `layernorm`, либо без
- **Инициализация** — orthogonal, gain=0.5
- **Dropout** — опциональный слой после каждой activation

Каждый скрытый слой: `Linear → Norm → Activation → Dropout(опционально)`.

Выход декодера — сырые логиты (без sigmoid), подаются в `BCEWithLogitsLoss`.

```
Input (D) → ... → Bottleneck (B) → ... → Output (D)
       └─ encoder ─┘          └─ decoder ─┘
```

#### Два способа задания архитектуры

Форма слоёв задаётся через sweep или вручную:

| Форма | Описание | Используется |
|-------|----------|-------------|
| **Rectangular** (по умолчанию) | `[D] + [H]×n + [B] + [H]×n + [D]` — все скрытые слои одинаковой ширины H | `make_rectangular()` |
| **Pyramid** | `[D] → h₁ → ... → hₙ → [B] → hₙ → ... → h₁ → [D]` — сужающаяся к центру | `make_pyramid()` |

Где:
- `D` = `seq_len × 21` (input_dim), например 128×21 = 2688
- `B` = `bottleneck` (по умолчанию = seq_len)
- `H` = `hidden_dim` = `D × b` (b — коэффициент ширины)
- `n` = количество скрытых слоёв с каждой стороны

#### Быстрый пример: создать модель

```python
from model import Autoencoder

# 15-layer autoencoder: 2688 → ... → 128 → ... → 2688
sizes = [2688, 5376, 10752, 5376, 5376, 128, 5376, 5376, 10752, 5376, 2688]
model = Autoencoder(sizes, activation='silu', normalization='batchnorm', dropout=0.1)

# Энкодинг
z = model.encode(x)  # z.shape = (batch, 128)

# Декодинг (возвращает логиты — нужен sigmoid для вероятностей)
logits = model.decode(z)
```

---

## Data Pipeline (`data.py`)

### Кодировка Unicode-21

Каждый символ ∈ [U+0000, U+1FFFFF] представляется как 21 бит:
- `ord(char)` → 21-битное целое → вектор из 21 нуля/единицы (старший бит первый)

### SlidingWindowDataset

Окна со **stride=1 символ** — каждый символ появляется в `seq_len` разных окнах.
Без перекрытия при большом `seq_len` возникала бы «голодовка данных» (мало сэмплов).

Реализация: `torch.as_strided` на float32-тензоре — окна это view, без копирования.

### Двухуровневое кэширование

| Уровень | Хранение | Размер (~50M символов) |
|---------|----------|------------------------|
| Диск | uint8 packed (`data/cache/full_bits.u8`) | ~0.13 GB |
| RAM | float32 unpacked | ~4.2 GB |

При первом запуске: биты упаковываются в uint8 на диск, при загрузке — распаковываются в float32.
В RAM нужно float32 для `as_strided` (uint8 не поддерживает произвольные страйды).

Автоматическая миграция со старого float32-кэша.

---

## Файлы

```
enc_dec/
├── model.py         # Autoencoder (nn.Module): encoder, decoder, init, dropout
├── configs.py        # PrimaryConfig (датакласс) — все гиперпараметры
├── data.py           # Загрузка, Unicode-21, SlidingWindowDataset, кэш
├── trainers.py       # Тренировочный цикл: AMP, early stopping, GPU-safe прерывание
├── logger.py         # CSV-логгер и resume-утилиты
├── autoencoder.py    # CLI-тренировка: python autoencoder.py
├── infer_test.py     # Интерактивная инференс-консоль
├── sweep.py          # CLI sweep-раннер: grid, binary, grid×binary
├── sweep_lib.py      # Разрешение архитектуры, train_one(), CSV-логирование
├── sweep_config.py   # SweepConfig, ModelConfig, TrainingConfig (датаклассы)
├── SWEEP.md          # Документация по sweep-системе
├── README.md         # ← этот файл
├── configs/          # 28 JSON-конфигов для разных sweep-экспериментов
│   ├── ratio_20m.json
│   ├── ratio_40m.json
│   ├── ratio_80m.json
│   ├── width_sweep.json
│   ├── batch_sweep.json
│   ├── norm_*.json           # Абляция нормализации (norm_bottleneck × norm_last)
│   ├── norm_comp_*.json      # Сравнение batchnorm vs layernorm
│   ├── pyramid_*.json        # Pyramid-архитектура
│   ├── rect_binary_*.json    # Бинарный поиск по n
│   └── binary_search.json    # Grid seq_len × binary n
├── sessions/        # Чекпоинты (.pth) и CSV-логи (автосоздаётся)
├── data/
│   ├── dataset/     # .txt файлы (~50M символов русской прозы)
│   └── cache/       # full_bits.u8 (автосоздаётся)
└── archive/         # Старые файлы и RESEARCH
```

---

## Быстрый старт

### Тренировка одной модели

```bash
# Из командной строки — простая тренировка на 30 эпох
python autoencoder.py
```

Конфигурация по умолчанию (в `configs.py`):
- `seq_len=128`, `input_dim=2688`, `bottleneck=128`
- `lr=5e-5`, `batch_size=1024`, 11-layer архитектура
- `BCEWithLogitsLoss`, AdamW fused, AMP

### Sweep нескольких моделей

См. [`SWEEP.md`](SWEEP.md) — полная документация.

```bash
# Grid: перебрать n=2,4,6,8,10, b подобрать под 40M параметров
python sweep.py grid --vary n=2,4,6,8,10 --solve b --budget 40M

# Из конфига
python sweep.py run --config configs/ratio_40m.json
```

### Инференс: протестировать обученные модели

```bash
python infer_test.py
# или на GPU:
python infer_test.py --gpu
```

Интерактивная консоль для проверки качества реконструкции:

```
  #  n_params  seq_len  n_hidden  folder                  file
  0  160M      128      4         ratio40                 2688_7644_7644_128_sweep_n2_best.pth
  1  160M      128      6         ratio40                 2688_5376_5376_128_sweep_n6.pth

> 0                  # загрузить модель #0
Loaded #0: s128, 160,000,000 params

> Привет, мир!      # реконструировать текст
Привет, мир!

> random             # случайное окно из датасета
@12345678: 'Анна Каренина...'
Анна Каренина...

> full 0             # первые 20 окон подряд
@0: '— Ну что, приехали?' → '— Ну что, приехали?'
...

> val 0 1            # быстрая валидация моделей #0 и #1
  #0 (s128, 160M)... val=0.031245
  #1 (s128, 160M)... val=0.030891

> q                  # выход
```

**Команды `infer_test.py`:**

| Команда | Описание |
|---------|----------|
| `<#>` / `load <#>` | Загрузить модель по номеру |
| `val <#> <#> ...` | Быстрая валидация на 1% датасета (без загрузки) |
| `random` / `r` | Случайное окно из датасета |
| `full [pos]` | Первые 20 окон подряд с позиции `pos` |
| `<любой текст>` | Реконструировать введённый текст |
| `q` / `quit` | Выход |

---

## Детали тренировки

### Функция потерь

`BCEWithLogitsLoss` — каждый из 21 бита на символ классифицируется независимо.
Декодер выдаёт **логиты** (без sigmoid), лосс сам применяет sigmoid внутри.

### Управление LR

Поддерживаются три стратегии (поле `lr_scheduler` в конфиге):

| Стратегия | Описание |
|-----------|----------|
| `cosine` | Линейный warmup → CosineAnnealingLR |
| `plateau` | Линейный warmup → ReduceLROnPlateau |
| `""` (пусто) | Без шедулера, константный LR |

При рестарте warmup пропускается — LR уже на рабочем уровне.

### Early stopping

По валидационной ошибке. Если `patience` чекпоинтов без улучшения — тренировка останавливается.
Лучшая модель сохраняется как `*_best.pth`.

При запуске через sweep `no_val=True` — early stopping отключён (быстрее, но без защиты от переобучения).

### Mixed precision (AMP)

Автоматически включается на CUDA: `autocast(bfloat16)` + `GradScaler`.
Даталоадер использует `pin_memory=True` + `non_blocking=True` для асинхронной передачи.

---

## GPU-безопасность

Система защиты от GPU ERR! (проблема RTX 3070 Mobile):

- **Signal handlers** — только устанавливают флаг, **никогда не вызывают CUDA** из обработчика
- **Graceful cleanup** — `torch.cuda.synchronize()` + `empty_cache()` при нормальном завершении
- **`atexit` handler** — страховка: даже при падении без обработчика
- **OOM protection** — ловится, GPU чистится, модель пропускается в sweep
- **Crash-only прерывания** — безопасные точки между батчами, CUDA не прерывается в середине операции

> Если GPU ушёл в состояние ERR!, поможет только `sudo reboot` + физическое отключение питания.
> Подробнее — [`MEMORY.md`](../MEMORY.md), секция «GPU — RTX 3070 Mobile».

---

## Конфигурация

См. `TrainingConfig` и `ModelConfig` в [`sweep_config.py`](sweep_config.py) —
расширенный набор полей для sweep-экспериментов (weight decay, warmup, оптимизатор и др.).
