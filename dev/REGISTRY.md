# REGISTRY.md — Реестр, дедупликация, sessions/

> Релевантные файлы: `registry/db.py`, `registry/schema.py`, `registry/fingerprint.py`, `orchestration/run.py`, `orchestration/sweep.py`

## Registry (SQLite)

`registry.db` — единый источник правды. Хранит fingerprint каждой обученной модели и позволяет дедуплицировать запуски.

### Fingerprint

Два хеша, определяющих уникальность модели:

1. **`arch_fingerprint(sizes, mc)`** — SHA256 от архитектуры: `sizes` (массив размеров слоёв) + model_config (без seq_len, bottleneck — они влияют на размеры)
2. **`training_hash(tc)`** — SHA256 от тренировочного конфига (без `num_workers`, `checkpoint_interval`, `early_stop_patience`, `target_samples`)

`UNIQUE(architecture_fp, training_hash)` в БД.

### Дедупликация

`Run.find_or_create()`:
- Нашёл run → `status=done` + `total_samples >= target_samples` → **пропускает**
- Нашёл run → `total_samples < target_samples` → **возобновляет** с чекпоинта
- Не нашёл → **создаёт новый**

Это позволяет:
- Перезапускать sweep — прерванные модели продолжаются
- Менять JSON-конфиг и запускать заново — старые не переобучаются
- Разные sweep'ы переиспользуют одинаковые модели

**Важно:** `target_samples` не входит в fingerprint. Если меняешь `target_samples`, fingerprint тот же → существующий run дотренируется до нового target.

### Fine-tune дедупликация

`pretrain_run_id` входит в `training_hash` → одинаковые архитектуры с разным претрейном считаются разными ранами.

### Таблицы

| Таблица | Содержимое |
|---------|-----------|
| `architectures` | fingerprint, n_params, seq_len, shape, config_json |
| `runs` | id, architecture_fp, training_hash, final_train_loss, total_samples, status |
| `experiments` | id, name, strategy, vary_param, config_json |
| `experiment_runs` | M:N связь экспериментов и запусков |

Views: `run_summary`, `experiment_results`.

---

## Структура sessions/

```
sessions/
├── runs/
│   ├── {hash}-{model_name}/
│   │   ├── model.pth / best.pth       # чекпоинты
│   │   ├── model.opt / model.sch      # оптимизатор / шедулер
│   │   ├── model.step_sch             # step-шедулер
│   │   ├── meta.json                  # конфиг модели
│   │   ├── result.json                # финальные метрики
│   │   ├── log.csv                    # метрики по шагам
│   │   └── train.log                  # текстовый лог
│   └── {hash} → {hash}-{model_name}   # symlink (backward compat)
├── experiments/
│   └── {exp_name}/
│       ├── config.json                # копия SweepConfig
│       └── summary.csv
├── registry.db                        # SQLite (в .gitignore)
├── plots/                             # output plot-скриптов
├── _overfit/                          # overfit-тесты
├── _ad_hoc_artifacts/                 # легаси/ручные запуски
└── archive/                           # сломанные/заархивированные раны
```

**Имена run-директорий:** `{12-char-hash}-{model_name}`, где model_name генерится из архитектуры, например `rect_s128_n6_b2.2013`.

## Ключевые модули

| Файл | Ответственность |
|------|----------------|
| `registry/db.py` | `get_db()`, CRUD операции |
| `registry/schema.py` | SQL-схема таблиц и views |
| `registry/fingerprint.py` | `arch_fingerprint()`, `training_hash()` |
| `orchestration/run.py` | `Run.find_or_create()`, `run.execute()` |
| `orchestration/sweep.py` | `Sweep.run()` — логика grid/binary |
| `orchestration/workspace.py` | `Workspace` — управление директориями |
| `orchestration/paths.py` | Пути к session-директориям |
