# CONFIG.md — JSON-конфиги и все поля

Конфигурация задаётся через JSON-файл. Три секции: `model`, `training`, `sweep`, плюс `name` и `output`.

## Полный формат (справочник всех полей)

### `model` — архитектура

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `seq_len` | int | 128 | Длина окна в символах |
| `bottleneck` | int\|null | null (= seq_len) | Размер бутылочного горла |
| `n` | int\|null | null | Скрытых слоёв (симметрично) |
| `enc_n` | int\|null | null | Слоёв в энкодере |
| `dec_n` | int\|null | null | Слоёв в декодере |
| `b` | float\|null | null | Коэффициент ширины (null → auto из бюджета) |
| `shape` | str | `"rectangular"` | `rectangular` / `pyramid` / `interleaved` / `trapezoid` |
| `activation` | str | `"silu"` | `silu` / `relu` / `gelu` / `leaky_relu` |
| `normalization` | str | `"batchnorm"` | `batchnorm` / `layernorm` / `rmsnorm` / `none` |
| `init` | str | `"orthogonal"` | `orthogonal` / `xavier` / `kaiming` |
| `init_gain` | float | 1.0 | Gain для инициализации |
| `dropout` | float | 0.0 | Dropout между слоями |
| `residual` | bool | false | Residual connections |
| `residual_norm` | str\|null | null | `"pre"` — pre-norm вариант |
| `norm_bottleneck` | bool | false | Norm на bottleneck |
| `norm_last` | bool | false | Norm на выходе декодера |
| `trapezoid_alpha` | float\|null | null | Отклонение trapezoid от base |
| `vae` | bool | false | VAE-режим (μ/logvar head) |

### `training` — обучение

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `target_samples` | int | 5_000_000 | Сколько семплов обучить |
| `batch_size` | int | 256 | Размер батча |
| `lr` | float | 0.001 | Learning rate |
| `grad_clip` | float | 1.0 | Gradient clipping |
| `scheduler` | str | `"onecycle"` | `onecycle` / `plateau` / `cosine` / `greedy` / `greedy_simple` / `greedy_grad` / `none` |
| `warmup_fraction` | float | 0.02 | Доля warmup (для schedulers с поддержкой) |
| `optimizer` | str | `"adamw_fused"` | `adamw_fused` / `adamw` / `lion` / `sophia` / `sgd` / `nag` |
| `weight_decay` | float | 0.01 | Weight decay |
| `decay_linear_only` | bool | false | Weight decay только на Linear |
| `early_stop_patience` | int | 20 | Валидаций без улучшения до остановки |
| `train_ratio` | float | 0.999 | Доля train (остальное val) |
| `num_workers` | int | 2 | DataLoader workers |
| `val_interval` | int | 100_000 | Семплов между валидациями |
| `checkpoint_interval` | int | 1_000_000 | Семплов между сохранениями |
| `use_tf32` | bool | true | TF32 на Ampere+ |
| `compile` | bool | false | torch.compile |
| `compile_mode` | str | `"default"` | Режим compile |
| `no_val` | bool | false | Отключить валидацию |
| `noise_prob_min` | float | 0.0 | Нижняя граница шума (0..1) |
| `noise_prob_max` | float | 0.0 | Верхняя граница (0 → disabled) |
| `noise_std_min` | float | 3.0 | Нижняя граница σ шума |
| `noise_std_max` | float | 3.0 | Верхняя граница σ |
| `noise_strategy` | str | `"linear"` | `linear` / `uniform` |
| `noise_stride` | int | 256 | Шаг интерполяции для linear |
| `pretrain_run_id` | str | `""` | Run ID донора для fine-tune |
| `vae_beta` | float | 1.0 | β для KL-терма |

### `sweep` — перебор

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `strategy` | str | `"grid"` | `grid` / `binary` |
| `vary` | str | — | Перебираемый параметр |
| `values` | list | — | Значения для grid / [lo, hi] для binary |
| `binary_on` | str\|null | null | Вложенный binary-поиск по параметру |
| `binary_range` | [float, float]\|null | null | Диапазон для вложенного binary |
| `solve` | str\|null | null | `"b"` / `"n"` — авто-подбор свободного параметра |
| `budget` | int\|null | null | Бюджет параметров |
| `fixed` | object | {} | Фиксированные параметры |
| `no_val` | bool | false | Отключить валидацию для sweep |

### `output` — пути

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `workspace` | str | `"sessions/sweep"` | Куда писать run-директории |
| `sweep_log` | str | `"sessions/global.csv"` | CSV-лог sweep'а |
| `device` | str | `"auto"` | `auto` / `cuda` / `cpu` |

### `name`

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `name` | str | — | Имя эксперимента |

---

## Пример конфига

```json
{
  "name": "noise_sweep_n6",
  "model": {
    "seq_len": 128,
    "shape": "rectangular",
    "activation": "silu",
    "normalization": "batchnorm"
  },
  "training": {
    "target_samples": 4000000,
    "batch_size": 256,
    "lr": 0.001,
    "scheduler": "onecycle",
    "optimizer": "adamw_fused",
    "early_stop_patience": 20
  },
  "sweep": {
    "strategy": "grid",
    "vary": "noise_prob",
    "values": [0.025, 0.05, 0.1, 0.15, 0.2, 0.25],
    "fixed": {"n": 6},
    "solve": "b",
    "budget": 4000000
  },
  "output": {
    "workspace": "sessions/noise_sweep_n6"
  }
}
```

---

## `--override` — переопределение из CLI

```bash
# Меняем model.seq_len и sweep.strategy, не редактируя JSON
bin/enc-dec sweep run --config configs/rect_sweep_384m.json \
  --override model.seq_len=64 sweep.strategy=binary

# Несколько оверрайдов
bin/enc-dec train --config configs/n8_rect_bn160_50M.json \
  --override training.lr=0.0001 training.scheduler=greedy

# Шум через оверрайд
bin/enc-dec sweep run --config configs/rect_sweep_384m.json \
  --override model.residual=true model.residual_norm=pre training.noise_prob_min=0.25

# VAE
bin/enc-dec train --config config.json \
  --override model.vae=true training.vae_beta=0.5
```

Dotted-нотация: `секция.поле=значение`. Значение парсится по типу существующего поля.
