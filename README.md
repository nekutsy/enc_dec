# enc_dec — текст в латентный вектор через автоэнкодер

Симметричный автоэнкодер, сжимающий текст (набор символов Unicode) в компактный
латентный вектор и восстанавливающий его обратно. Обучается как бинарный
классификатор на 21 бите каждого Unicode codepoint.

**Сжатие:** 21:1 (128 символов × 21 бит → 128-мерный латентный вектор).

---

## Документация

| Файл | Содержание |
|------|-----------|
| **[USAGE.md](USAGE.md)** | Полная инструкция по использованию: CLI, команды, JSON-конфиги |
| **[SWEEP.md](SWEEP.md)** | Sweep-система: стратегии, параметры, дедупликация |

---

## Архитектура

Симметричный полносвязный автоэнкодер с настраиваемыми:
- **Глубина** — задаётся массивом `layer_sizes`, середина = bottleneck
- **Асимметрия** — можно задать разное количество слоёв в энкодере и декодере (`--enc-n`, `--dec-n`)
- **Форма** — rectangular, pyramid, interleaved, trapezoid
- **Активация** — `silu`, `relu`, `gelu`, `leaky_relu`
- **Нормализация** — `batchnorm` (BatchNorm1d), `layernorm`, `rmsnorm`, `none`
- **Инициализация** — orthogonal, xavier, kaiming, настраиваемый gain
- **Dropout** — опциональный
- **Residual connections** — classic (post-norm) или pre-norm

Каждый скрытый слой: `Linear → Norm → Activation → Dropout`.

Выход декодера — сырые логиты (без sigmoid), подаются в `BCEWithLogitsLoss`.

```
Input (D) → ... → Bottleneck (B) → ... → Output (D)
       └─ encoder ─┘          └─ decoder ─┘
```

Где `D = seq_len × 21` (например 128 × 21 = 2688), `B = bottleneck`.

### Формы архитектуры

| Форма | Описание |
|-------|----------|
| **Rectangular** | `[D] + [H]×n + [B] + [H]×n + [D]` — скрытые слои одинаковой ширины |
| **Pyramid** | `[D] → h₁ → … → hₙ → [B] → hₙ → … → h₁ → [D]` — сужающаяся к центру |
| **Interleaved** | Wide-слои чередуются с narrow для создания неоднородной структуры |
| **Trapezoid** | Линейная интерполяция ширины слоёв с α-отклонением |

---

## Структура проекта

```
enc_dec/
├── bin/
│   └── enc-dec              # Единая точка входа CLI
├── cli/                     # CLI-скрипты (train, sweep, infer, overfit, status, main)
├── model/                   # Autoencoder (nn.Module), архитектурные билдеры
├── training/                # Цикл обучения, шаг, шедулеры, чекпоинты
├── orchestration/           # Run (одиночное обучение), Sweep (сетка/бинарный), Workspace
├── registry/                # SQLite-реестр экспериментов и запусков
├── inference/               # Инференс: сканер моделей, API, REPL
├── experiment/              # Конфиги (датаклассы), runtime context, пресеты
├── encoding/                # Unicode-21 кодировка
├── core/                    # Базовые типы
├── configs/                 # Готовые SweepConfig JSON-файлы
├── scripts/                 # Утилиты (plot, resume, migrate)
├── data/
│   ├── dataset/             # .txt файлы (~50M символов русской прозы)
│   └── cache/               # full_bits.u8 (автосоздаётся)
├── sessions/                # Чекпоинты, CSV-логи, реестр
├── USAGE.md                 # Инструкция по использованию
├── SWEEP.md                 # Sweep-специфика
└── README.md                # ← этот файл
```

---

## Data Pipeline

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

---

## Быстрый старт

```bash
# Проверить архитектуру
bin/enc-dec overfit --seq-len 128 --n 8 --budget 384M

# Обучить одну модель
bin/enc-dec train --n 3 --budget 160M --samples 50M

# Запустить sweep из конфига
bin/enc-dec sweep run --config configs/noise_sweep.json

# Посмотреть прогресс
bin/enc-dec status

# Протестировать модель
bin/enc-dec infer --gpu
```

Полная инструкция — **[USAGE.md](USAGE.md)**.

---

## Детали тренировки

### Функция потерь

`BCEWithLogitsLoss` — каждый из 21 бита классифицируется независимо.

### LR-шедулеры

`onecycle` (по умолчанию), `plateau`, `cosine`, `greedy` (адаптивный zeroth-order с пробами), `greedy_simple`, `greedy_grad`, `none`.

### Оптимизаторы

`adamw_fused` (fused CUDA), `adamw`, `lion` (EvoLved Sign Momentum), `sophia`, `sgd`, `nag`.

### Mixed precision

Автоматически на CUDA: `autocast(bfloat16)` + `GradScaler`.

---

## Результаты исследований

См. [`archive/RESEARCH_2026-06-18.md`](archive/RESEARCH_2026-06-18.md) — актуальные результаты sweep-экспериментов.
