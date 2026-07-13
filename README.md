# enc_dec — текст в латентный вектор через автоэнкодер

Симметричный автоэнкодер, сжимающий текст в компактный латентный вектор и восстанавливающий обратно.
Обучается как бинарный классификатор на 21 бите Unicode (BCEWithLogitsLoss).

**Сжатие:** 21:1 (128 символов × 21 бит → 128-мерный латент).

---

## Документация

### Для пользования (`doc/`)

| Файл | Содержание |
|------|-----------|
| **[USAGE.md](doc/USAGE.md)** | CLI-команды: train, infer, overfit, lr-find, plot, resume |
| **[CONFIG.md](doc/CONFIG.md)** | JSON-формат, все поля ModelConfig/TrainConfig/SweepConfig, `--override` |
| **[SWEEP.md](doc/SWEEP.md)** | Стратегии sweep (grid, binary, binary_on), solve, vary, presets |
| **[FEATURES.md](doc/FEATURES.md)** | VAE, denoising, fine-tune, асимметрия |
| **[GPU.md](doc/GPU.md)** | Правила безопасности GPU ⚠️ критично |

### Для разработки (`dev/`)

| Файл | Содержание |
|------|-----------|
| **[ARCHITECTURE.md](dev/ARCHITECTURE.md)** | Устройство автоэнкодера: формы, слои, solve, подсчёт параметров |
| **[TRAINING.md](dev/TRAINING.md)** | Тренировочный цикл, шедулеры, оптимизаторы, mixed precision |
| **[REGISTRY.md](dev/REGISTRY.md)** | SQLite-реестр, fingerprint, дедупликация, sessions/ |
| **[DATA.md](dev/DATA.md)** | Unicode-21, SlidingWindowDataset, кэширование |

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

---

## Архитектура (кратко)

Симметричный полносвязный автоэнкодер: `Input → ... → Bottleneck → ... → Output`.
Каждый слой: `Linear → Norm → Activation → Dropout`.

4 формы: `rectangular`, `pyramid`, `interleaved`, `trapezoid`.
Активации: SiLU (default), ReLU, GELU, LeakyReLU.
Нормализация: BatchNorm1d, LayerNorm, RMSNorm, none.

Подробно: [dev/ARCHITECTURE.md](dev/ARCHITECTURE.md).

---

## Структура проекта

```
enc_dec/
├── bin/enc-dec              # Единая точка входа CLI
├── cli/                     # CLI-обработчики
├── model/                   # Autoencoder, архитектурные билдеры, формы
├── training/                # Цикл обучения, шедулеры, оптимизаторы
├── orchestration/           # Run, Sweep, Workspace
├── registry/                # SQLite-реестр
├── inference/               # Сканер моделей, API, REPL
├── experiment/              # Датаклассы конфигов, пресеты
├── encoding/                # Unicode-21
├── data/                    # Датасет + кэш
├── configs/                 # Готовые SweepConfig JSON (~30 шт.)
├── sessions/                # Чекпоинты, логи, registry.db
├── doc/                     # Документация «для пользования»
├── dev/                     # Документация «для разработки»
└── scripts/                 # Утилиты (plot, migrate, resume)
```

---

## Результаты исследований

См. [`archive/RESEARCH_2026-06-18.md`](archive/RESEARCH_2026-06-18.md) — актуальные результаты sweep-экспериментов.
