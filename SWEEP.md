# SWEEP.md — Свипы и стратегии

> Базовая инструкция по использованию и формат JSON-конфигов — в **[USAGE.md](USAGE.md)**.

Sweep-система перебирает параметры архитектуры/обучения по заданной стратегии, используя Registry для дедупликации: если модель с такими же параметрами уже обучена — пропускает.

## Команды

```bash
# Из JSON-конфига (основной способ)
enc-dec sweep run --config configs/noise_sweep.json

# С оверрайдом — не редактируя JSON
enc-dec sweep run --config configs/rect_sweep_384m.json \
  --override model.seq_len=64 sweep.strategy=binary

# Shorthand: grid
enc-dec sweep grid --vary n=2,4,6,8,10 --solve b --budget 40M

# Shorthand: binary
enc-dec sweep binary --vary n --range 2 16 --solve b --budget 40M
```

Оверрайды через `--override`: dotted-нотация (`model.seq_len=64`, `training.lr=0.002`, `sweep.solve=n`). Значение подставляется по типу существующего поля (если поле int — парсится как int, float → float, иначе str).

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

Перебирает все `values` по порядку. С `solve: "b"` для каждого n подбирает коэффициент ширины `b` под заданный бюджет параметров.

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

`values` — границы диапазона `[lo, hi]`. Алгоритм:
1. Пробует границы (lo, hi)
2. Смотрит у какой из них loss лучше
3. Биссектирует между лучшей и второй лучшей
4. Повторяет до сходимости (соседние значения уже проверены)

### `grid` с вложенным бинарным поиском

Можно задать `binary_on` для grid'а — каждый кандидат проверяется бинарным поиском по вложенному параметру:

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

## Параметр `solve`

Управляет тем, как заполняются свободные параметры архитектуры:

| solve | Описание |
|-------|----------|
| `"b"` | По `n` + `budget` подбирает коэффициент ширины `b` |
| `"n"` | По `b` + `budget` подбирает количество слоёв `n` |
| `null` | Оба параметра заданы вручную (через `fixed` или values) |

## Параметр `vary`

Что перебирается. Допустимые значения:

**Архитектурные** (`MODEL_LEVEL_VARY`):
- `n` — количество скрытых слоёв
- `b` — коэффициент ширины
- `bottleneck` — размер бутылочного горла
- `normalization`, `activation`, `dropout`, `norm_bottleneck`, `norm_last`, `trapezoid_alpha`

**Тренировочные** (`TRAIN_LEVEL_VARY`):
- `lr`, `scheduler`, `grad_clip`, `optimizer`, `weight_decay`, `batch_size`, `num_workers`, `noise_prob`, `noise_std`

Если `vary` — training-level параметр, в `fixed` должны быть заданы `n` и/или `b`.

## Архитектурные формы

Задаются через `model.shape`:

| `shape` | Описание | Примечание |
|---------|----------|------------|
| `rectangular` | `[D] + [H]×n + [B] + [H]×n + [D]` | Все скрытые слои одной ширины |
| `pyramid` | Сужающаяся к центру | `solve=d` подбирает градиент сужения |
| `interleaved` | Wide-слои чередуются с narrow | Решает ту же задачу, что и pyramid |
| `trapezoid` | Линейная интерполяция ширины | `trapezoid_alpha` — отклонение от base |

## Готовые конфиги

```
configs/
├── rect_sweep_384m.json            # n=2..8, solve=b, budget=384M
├── trapezoid_sweep_384m.json        # То же с trapezoid shape
├── pyramid_sweep_384m.json          # То же с pyramid shape
├── interleaved_sweep_384m.json      # То же с interleaved shape
├── noise_sweep.json                 # noise_prob grid
├── bottleneck_sweep_4M.json         # bottleneck размеры
├── bs_sweep_4M_noise0025.json       # batch sizes
└── ...
```

## Дедупликация и кэширование

Registry (SQLite `sessions/registry.db`) хранит fingerprint каждой обученной модели (архитектура + тренировочный конфиг). При повторном запуске того же sweep'а:

1. `Run.find_or_create()` проверяет Registry
2. Если `status=done` + `total_samples >= target_samples` → пропускает
3. Если частично обучена (меньше семплов) → возобновляет
4. Если не найдена → создаёт новую

Это позволяет:
- Перезапускать sweep без потери прогресса (прерванные модели продолжаются)
- Менять JSON-конфиг и запускать заново — старые модели не переобучаются
- Разные sweep'ы переиспользуют одинаковые модели (например, noise_sweep и rect_sweep с пересекающимися параметрами)

## Сводка по аргументам sweep

| Флаг | Описание |
|------|----------|
| `--config` | JSON-конфиг (для `run`) |
| `--override` | `путь=значение`, можно несколько |
| `--vary` | `n=2,4,6` (grid) или имя параметра (binary) |
| `--range` | `lo hi` (binary) |
| `--solve` | `b` или `n` |
| `--fixed` | `n=7 b=4` — фиксированные параметры |
| `--budget` | `40M` |
| `--seq-len` | 32 |
| `--lr` | 0.001 |
| `--scheduler` | onecycle |
| `--target-samples` | 5M |
| `--workspace` | sessions/sweep |
| `--device` | auto |
