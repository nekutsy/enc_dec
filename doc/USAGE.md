# USAGE.md — CLI-команды

> Конфигурация (JSON-формат, все поля) → [CONFIG.md](CONFIG.md)
> Фичи (VAE, шум, fine-tune, асимметрия) → [FEATURES.md](FEATURES.md)

## Точка входа

```bash
bin/enc-dec <команда> [опции]
```

`bin/enc-dec <cmd> --help` — справка по любой команде.

| Команда | Назначение |
|---------|-----------|
| `status` | Просмотр реестра экспериментов и запусков |
| `train` | Обучение одной модели |
| `sweep` | Grid/binary search по параметрам |
| `infer` | Интерактивная инференс-консоль |
| `overfit` | Overfit-тест на одном батче |
| `lr-find` | LR range test |
| `plot` | Графики по логам |
| `resume` | Дослать все запуски до N семплов |

---

## `status` — реестр

```bash
bin/enc-dec status                 # обзор: сколько экспериментов/запусков, последние done
bin/enc-dec status --runs          # все запуски с деталями
bin/enc-dec status --experiments   # список экспериментов
bin/enc-dec status --exp NAME      # детали эксперимента
bin/enc-dec status --run ID        # детали запуска (первые 6+ символов)
```

---

## `train` — одиночная модель

```bash
# Минимальный запуск
bin/enc-dec train --n 3 --budget 160M --samples 50M

# Из готового JSON-конфига
bin/enc-dec train --config configs/n8_rect_bn160_50M.json

# С переопределением параметров
bin/enc-dec train --n 2 --budget 160M --samples 120M --lr 0.002 --scheduler plateau
```

**Авто-возобновление:** повторный запуск подхватывает чекпоинт и продолжает.
**Дедупликация:** если модель с теми же параметрами уже обучена до `target_samples` — пропустит.

### Основные аргументы

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--n` | обязателен¹ | Скрытых слоёв с каждой стороны (симметрично) |
| `--enc-n` | = n | Слоёв в энкодере |
| `--dec-n` | = enc_n | Слоёв в декодере |
| `--budget` | обязателен¹ | Целевое количество параметров (`160M`, `40M`) |
| `--samples` | `50M` | Сколько семплов обучить |
| `--seq-len` | 128 | Длина окна в символах |
| `--bottleneck` | = seq_len | Размер бутылочного горла |
| `--b` | — | Коэффициент ширины (вместо auto из бюджета) |
| `--shape` | rectangular | Форма архитектуры |
| `--lr` | 0.001 | Learning rate |
| `--scheduler` | onecycle | Шедулер |
| `--optimizer` | adamw_fused | Оптимизатор |
| `--activation` | silu | Функция активации |
| `--normalization` | batchnorm | Тип нормализации |
| `--vae` | false | VAE-режим |
| `--vae-beta` | 1.0 | β вес KL-терма |
| `--noise-prob` | — | Доля зашумлённых символов (0..1) |
| `--noise-prob-min/max` | 0.0 | Per-batch диапазон шума |
| `--pretrain-from` | — | Run ID донора для fine-tune |
| `--device` | auto | auto / cuda / cpu |
| `--batch-size` | 256 | Размер батча |
| `--no-val` | false | Отключить валидацию |

¹ Не нужны при использовании `--config`.

Полный список полей с описанием: [CONFIG.md](CONFIG.md).

---

## `sweep` — перебор параметров

```bash
# Из JSON-конфига (основной способ)
bin/enc-dec sweep run --config configs/noise_sweep.json

# С оверрайдом
bin/enc-dec sweep run --config configs/rect_sweep_384m.json \
  --override model.seq_len=64 sweep.strategy=binary

# Shorthand из CLI
bin/enc-dec sweep grid --vary n=2,4,6,8 --solve b --budget 40M
bin/enc-dec sweep binary --vary n --range 2 16 --solve b --budget 40M
```

Подробно о стратегиях, solve, vary, presets: [SWEEP.md](SWEEP.md).

---

## `infer` — интерактивная консоль

```bash
bin/enc-dec infer            # CPU
bin/enc-dec infer --gpu      # GPU
```

Команды внутри REPL:

| Команда | Описание |
|---------|----------|
| `<#>` | Загрузить модель по номеру |
| `enc <text\|random\|@pos>` | Закодировать → латент |
| `dec` | Декодировать сохранённый латент |
| `dec <values>` | Декодировать конкретный вектор |
| `z` | Показать латент |
| `random` / `r` | Случайное окно → реконструкция |
| `full [pos]` | 20 окон подряд |
| `sample [N]` | Генерация из приора (VAE only) |
| `val 0 1 3` | Валидация указанных моделей |
| `<любой текст>` | Прямая реконструкция |
| `q` | Выход |

Chain-команды: `dec enc random` — закодировать случайное окно и сразу декодировать.

---

## `overfit` — проверка архитектуры

```bash
bin/enc-dec overfit                                           # defaults
bin/enc-dec overfit --seq-len 128 --n 8 --budget 384M         # свои параметры
bin/enc-dec overfit --shape pyramid --seq-len 64 --n 4 --b 2.0
```

Вывод: `✅ CAN OVERFIT` / `⚠ OVERFITS POORLY` / `❌ CANNOT OVERFIT`.
Логи: `sessions/_overfit/<model_name>/log.csv`.

---

## `lr-find` — LR range test

```bash
bin/enc-dec lr-find --n 6 --budget 384M
bin/enc-dec lr-find --n 6 --budget 384M --lr-end 1.0
```

---

## `plot` — графики

```bash
bin/enc-dec plot runs                                          # все запуски
bin/enc-dec plot noise 0.025 RUN_ID_1 0.25 RUN_ID_2            # сравнение двух
```

Графики сохраняются в `sessions/plots/`.

---

## `resume` — дослать до N семплов

```bash
bin/enc-dec resume                         # до дефолтного target
RESUME_TARGET=20000000 bin/enc-dec resume   # свой target
```

Перезапускает обучение для всех моделей в `sessions/runs/`, подхватывая чекпоинты.
