# TRAINING.md — Тренировочный цикл

> Релевантные файлы: `training/loop.py`, `training/step.py`, `training/scheduler.py`, `training/optimizers.py`, `training/checkpoint.py`, `training/lr_finder.py`

## Общая схема

Обучение: `run.execute()` → `training/loop.py::train_loop()` — итеративный цикл с валидацией и чекпоинтами.

```
for batch in dataloader:
    step_train(batch)          # forward + backward + optimizer.step()
    if samples % val_interval == 0:
        validate()
    if samples % checkpoint_interval == 0:
        save_checkpoint()
```

## Функция потерь

`BCEWithLogitsLoss` — каждый из 21 бита Unicode-символа классифицируется независимо (бинарная классификация).

## Оптимизаторы (`training/optimizers.py`)

| Оптимизатор | Описание |
|-------------|----------|
| `adamw_fused` | Fused CUDA-реализация (default) |
| `adamw` | Стандартный AdamW |
| `lion` | EvoLved Sign Momentum |
| `sophia` | Second-order clipped |
| `sgd` | Классический SGD с momentum |
| `nag` | Nesterov Accelerated Gradient |

`decay_linear_only: true` — weight_decay только на Linear-слои (bias без decay).

## Шедулеры (`training/scheduler.py`)

### Per-step (каждый batch)

| Шедулер | Описание |
|----------|----------|
| `onecycle` | OneCycleLR с cosine anneal (default) |
| `cosine` | CosineAnnealingLR |
| `greedy_simple` | Immediate raise / cautious decrease; floor → reset cycle |
| `greedy_grad` | Gradient descent on LR с plateau escape |
| `none` | Константный LR |

### Per-checkpoint (на валидации)

| Шедулер | Описание |
|----------|----------|
| `plateau` | ReduceLROnPlateau |
| `greedy` | Zeroth-order GreedyLR с probing механизмом |

### Кастомные шедулеры

**GreedyLR** (`greedy`): zeroth-order оптимизация LR. Делает пробные шаги с увеличенным/уменьшенным LR, сравнивает loss, выбирает направление.

**GreedySimpleLR** (`greedy_simple`): упрощённый вариант — повышает LR при улучшении, понижает при ухудшении. При достижении floor сбрасывает цикл.

**GreedyGradLR** (`greedy_grad`): градиентный спуск по LR с escape из плато.

## Mixed Precision

Автоматически на CUDA: `autocast(bfloat16)` без GradScaler (bfloat16 не требует скейлинга, в отличие от float16).

TF32: `use_tf32: true` (по умолчанию) — ускорение на Ampere+.

## Чекпоинты (`training/checkpoint.py`)

```
sessions/runs/{id}-{name}/
├── model.pth       # Последний: веса + оптимизатор + шедулер
├── best.pth        # Лучший по val loss (только веса модели)
├── model.opt       # Состояние оптимизатора
├── model.sch       # Состояние checkpoint-шедулера
├── model.step_sch  # Состояние step-шедулера (greedy_simple, greedy_grad)
├── meta.json       # Конфиг модели
├── result.json     # Финальные метрики
├── log.csv         # Метрики по шагам (train_loss, val_loss, lr)
└── train.log       # Текстовый лог
```

## Overfit-тест (`training/lr_finder.py` + `cli/overfit_test.py`)

Проверяет, может ли архитектура заучить один батч.

```
✅ MODEL CAN OVERFIT    — loss падает до ~0 за разумное число шагов
⚠ MODEL OVERFITS POORLY — учится, но медленно
❌ MODEL CANNOT OVERFIT  — архитектура сломана
```

Логи: `sessions/_overfit/<model_name>/log.csv`.

## LR Range Test (`training/lr_finder.py`)

Классический LR range test: экспоненциальный рост LR от малого до `lr_end`, измерение loss. Полезно для выбора оптимального LR перед обучением.

## Ключевые модули

| Файл | Ответственность |
|------|----------------|
| `training/loop.py` | `train_loop()` — главный цикл |
| `training/step.py` | `step_train()` — один шаг (forward + backward + opt) |
| `training/scheduler.py` | Все шедулеры |
| `training/optimizers.py` | `create_optimizer()` — фабрика |
| `training/checkpoint.py` | `save_checkpoint()`, `load_checkpoint()`, `find_latest_checkpoint()` |
| `training/lr_finder.py` | `lr_find()` — LR range test, `run_overfit()` — overfit-тест |
