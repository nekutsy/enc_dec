# ARCHITECTURE.md — Устройство автоэнкодера

> Релевантные файлы: `model/architecture.py`, `model/autoencoder.py`, `model/shapes/*.py`

## Общая схема

Симметричный полносвязный автоэнкодер: текст → латентный вектор → текст.

```
Input (D) → ... → Bottleneck (B) → ... → Output (D)
       └─ encoder ─┘          └─ decoder ─┘

D = seq_len × 21   (например 128 × 21 = 2688)
B = bottleneck     (по умолчанию = seq_len)
```

Выход декодера — сырые логиты (без sigmoid), подаются в `BCEWithLogitsLoss`. Каждый из 21 бита Unicode классифицируется независимо.

## Слой

Каждый скрытый слой: `Linear → Norm → Activation → Dropout`

- **Norm:** BatchNorm1d / LayerNorm / RMSNorm / none
- **Activation:** SiLU (default), ReLU, GELU, LeakyReLU
- **Dropout:** опциональный

## Архитектурные формы

Формы задают распределение ширины слоёв. Реализованы в `model/shapes/*.py`.

### Rectangular
```
[D] + [H]×n + [B] + [H]×n + [D]
```
Все скрытые слои одинаковой ширины. Ширина H вычисляется из бюджета параметров через `solve(budget, n)`.

### Pyramid
```
[D] → h₁ → … → hₙ → [B] → hₙ → … → h₁ → [D]
```
Сужающаяся к центру. `solve=d` подбирает градиент сужения.

### Interleaved
Wide-слои чередуются с narrow, создавая неоднородную структуру. Решает ту же задачу, что и pyramid.

### Trapezoid
Линейная интерполяция ширины слоёв с α-отклонением от base. Параметр `trapezoid_alpha` контролирует степень отклонения.

## Параметр `solve`

Автоматический подбор архитектурных параметров под бюджет (`model/architecture.py`):

| solve | Описание |
|-------|----------|
| `"b"` | По `n` + `budget` → подбирает коэффициент ширины `b` |
| `"n"` | По `b` + `budget` → подбирает количество слоёв `n` |
| `null` | Оба параметра заданы вручную |

Алгоритм: бинарный поиск коэффициента/глубины до достижения целевого бюджета параметров с заданной точностью.

## Асимметрия (enc_n / dec_n)

Энкодер и декодер могут иметь разное количество слоёв. Реализовано в `model/architecture.py` и `model/autoencoder.py`:

- `--n` → `enc_n = dec_n = n` (симметрично)
- `--enc-n` / `--dec-n` → точный контроль
- Бюджет распределяется пропорционально количеству слоёв

## Подсчёт параметров

`model/architecture.py::count_params()`:
```
total = Σ(Linear_weights + Linear_bias + BatchNorm1d_weight + BatchNorm1d_bias)
```
Для LayerNorm/RMSNorm — аналогично, по 2 параметра на размерность.

## Дополнительные фичи архитектуры

### Residual connections

- `residual: true` — classic (post-norm): `(Linear → Norm → Act)(x) + x` (только при совпадении dims)
- `residual_norm: "pre"` — pre-norm: `x + (Linear → Act)(Norm(x))`

### VAE head

При `vae: true` bottleneck дополняется двумя линейными слоями:
- μ head: `Linear(B, B)`
- log σ² head: `Linear(B, B)`

`forward()` возвращает `(reconstruction, μ, log_var)`. `encode()` возвращает μ (детерминированно). `sample(N)` семплирует из N(0,I) и декодирует.

### Инициализация

`orthogonal` (по умолчанию), `xavier`, `kaiming` — с настраиваемым `init_gain`.

## Ключевые модули

| Файл | Ответственность |
|------|----------------|
| `model/autoencoder.py` | `Autoencoder(nn.Module)` — forward, encode, decode, sample |
| `model/architecture.py` | `build_sizes()`, `count_params()`, `solve_*()` |
| `model/shapes/rectangular.py` | Генератор размеров для rectangular |
| `model/shapes/pyramid.py` | Генератор для pyramid |
| `model/shapes/interleaved.py` | Генератор для interleaved |
| `model/shapes/trapezoid.py` | Генератор для trapezoid |
| `model/factory.py` | `create_autoencoder()` — фабрика моделей из ModelConfig |
