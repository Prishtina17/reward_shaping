# MARL Reward Shaping (SMAC + QMIX) - документация по коду и воспроизводимости

Этот репозиторий - инженерная реализация дипломного проекта: набор reward shaping окружений для SMAC/StarCraft II и инфраструктура для массовых воспроизводимых экспериментов на QMIX.

Файл `Диплом(2).docx` - "источник истины" по постановке задачи/метрикам/выводам. Этот README описывает именно код: где какая логика лежит, как она запускается, что пишется на диск и как это повторить.

## TL;DR: быстрые команды

- Один запуск: `python src/main.py --config=qmix --env-config=<env> with env_args.map_name=<map> seed=42 ...`
- Полный дипломный прогон: `bash run_all_shapings.sh`
- TensorBoard: `./run_tensorboard.sh` (логдир по умолчанию `results/tb_logs`)
- Чекпоинты: `results/models/<unique_token>/<t_env>/` (включить `save_model=True`)
- Оценка чекпоинта: `with checkpoint_path="results/models/<unique_token>" load_step=0 evaluate=True`

---

## 1. Карта репозитория (что за что отвечает)

```
.
├─ src/
│  ├─ main.py                         # CLI: сбор YAML + Sacred FileStorageObserver
│  ├─ config/
│  │  ├─ default.yaml                 # дефолты (логирование, save_model, пути, ...)
│  │  ├─ algs/qmix.yaml               # QMIX (runner=parallel, td_lambda, lr, ...)
│  │  └─ envs/melee_range_control_*.yaml
│  ├─ run/run.py                      # training loop: runner/buffer/learner, early-stop, чекпоинты
│  ├─ runners/                        # EpisodeRunner / ParallelRunner
│  ├─ envs/__init__.py                # REGISTRY: имя env -> класс окружения
│  └─ envs/starcraft/
│     ├─ StarCraft2Env.py             # базовая SMAC-среда + эпизодные диагностические метрики
│     ├─ melee_range_control_*.py     # shaping варианты (SB/AB/PBRS и комбинации)
│     └─ utils.py                     # общие функции shaping + ShapingMetrics
├─ run_all_shapings.sh                # дипломный протокол: карты x shapings + фиксированные интервалы
├─ run_tensorboard.sh                 # запуск TensorBoard для results/tb_logs
├─ plot_final_metrics.py              # финальные графики из final_tb_logs -> final_metrics
├─ final_tb_logs/                     # экспортированные логи TensorBoard (по картам)
├─ final_metrics/                     # готовые графики/таблицы (для диплома)
└─ MARL Reward Shaping/               # презентационный пакет (копии файлов под методичку)
```

### Поток выполнения (упрощенно)

```
run_all_shapings.sh
  -> src/main.py (Sacred + сбор YAML)
    -> src/run/run.py (runner + buffer + mac + learner)
      -> runner <-> env (StarCraft2Env или melee_range_control_*)
      -> Logger (TensorBoard + Sacred)
      -> checkpoints -> results/models/<unique_token>/<t_env>/
```

### 1.1 Трассируемость к диплому (что где в коде)

- Реализация shaping (SB/AB/PBRS и комбинации): `src/envs/starcraft/melee_range_control_*.py` + общие функции в `src/envs/starcraft/utils.py`.
- Единый протокол экспериментов (карты, интервалы, сохранения): `run_all_shapings.sh` + YAML-конфиги в `src/config/algs/` и `src/config/envs/`.
- Логирование и воспроизводимость (конфиги, артефакты, идентификаторы запусков): `src/main.py` (Sacred) и `src/run/run.py` (TensorBoard, checkpoints).
- Построение финальных графиков из TensorBoard логов: `plot_final_metrics.py` (читает `final_tb_logs/`, пишет в `final_metrics/`).

---

## 2. Точки входа и конфигурирование (Sacred/YAML)

### 2.1 Как собирается конфигурация

`src/main.py` загружает и рекурсивно сливает:

1) `src/config/default.yaml` (общие дефолты),
2) `src/config/algs/<algo>.yaml` (например, `qmix`),
3) `src/config/envs/<env>.yaml` (например, `melee_range_control_ab`),
4) overrides из CLI через `with key=value ...`.

Сохранение Sacred-логов включено по умолчанию: `results/sacred/<map>/<experiment_name>/...`.

### 2.2 Пример одиночного запуска

```bash
python src/main.py \
  --config=qmix \
  --env-config=melee_range_control_ab \
  with env_args.map_name=3s_vs_3z seed=42 save_model=True save_model_interval=50000
```

Практически полезные overrides:

- `seed=<int>`: `src/main.py` выставляет seed для numpy/torch и прокидывает его в `env_args.seed`.
- `training_stop_mode=win_rate training_win_rate_threshold=0.95`: ранняя остановка по win rate (см. `src/run/run.py`).
- `use_cuda=False`: CPU-режим.
- `checkpoint_path="results/models/<unique_token>" load_step=500000`: продолжить с чекпоинта.

---

## 3. Reward shaping: реализованные окружения и "где в коде математика"

### 3.1 Общие строительные блоки (общие для всех вариантов)

`src/envs/starcraft/utils.py`:

- `ring_function(d, center, half_width)` - "бублик": плато в sweet-spot и спад вне зоны.
- `compute_ring_bonus_from_state(...)` - state-based бонус по текущему состоянию.
- `ShapingMetrics` + `update_shaping_metrics(...)` - единый формат диагностик `shaping/*`.
- `enemy_damage_step(...)`, `ally_damage_step(...)`, `count_alive_*` - вспомогательные сигналы для логов.

Важно: в текущей версии `_is_melee(...)` в `src/envs/starcraft/utils.py` возвращает `True`, т.е. флаг `rc_melee_only` фактически не отфильтровывает юнитов по типу (шаги считаются относительно всех "живых" врагов). Если требуется строгая фильтрация melee/ranged, это место для доработки.

### 3.2 Таблица соответствия: env-config -> файл -> тип shaping

Все env-config регистрируются в `src/envs/__init__.py` и выбираются по `env:` из YAML.

| env-config | YAML | Python | Тип добавки |
| --- | --- | --- | --- |
| `sc2` | `src/config/envs/sc2.yaml` | `src/envs/starcraft/StarCraft2Env.py` | baseline |
| `melee_range_control_sb` | `src/config/envs/melee_range_control_sb.yaml` | `src/envs/starcraft/melee_range_control_sb.py` | SB (state ring bonus) |
| `melee_range_control_ab` | `src/config/envs/melee_range_control_ab.yaml` | `src/envs/starcraft/melee_range_control_ab.py` | AB (action bonus) |
| `melee_range_control_pb` | `src/config/envs/melee_range_control_pb.yaml` | `src/envs/starcraft/melee_range_control_pb.py` | PBRS (delta potential) |
| `melee_range_control_as` | `src/config/envs/melee_range_control_as.yaml` | `src/envs/starcraft/melee_range_control_as.py` | AB + SB |
| `melee_range_control_sp` | `src/config/envs/melee_range_control_sp.yaml` | `src/envs/starcraft/melee_range_control_sp.py` | SB + PBRS |
| `melee_range_control_ap` | `src/config/envs/melee_range_control_ap.yaml` | `src/envs/starcraft/melee_range_control_ap.py` | AB + PBRS |
| `melee_range_control_asp` | `src/config/envs/melee_range_control_asp.yaml` | `src/envs/starcraft/melee_range_control_asp.py` | AB + SB + PBRS |

### 3.3 Где именно добавляется shaping-награда (по коду)

- **SB**: считается внутри `reward_battle()` как `ring_function(d_min)` по живым союзникам и добавляется к базовой награде.
- **AB**: в `step()` до шага среды считается "правильность" действия относительно ближайшего врага и состояния оружия, затем добавка применяется в `reward_battle()`.
- **PBRS**: в `reward_battle()` считается потенциал `phi` (обычно средний ring), затем добавляется `rc_weight * (gamma * phi_curr - phi_prev)`.
- **Комбинации**: суммируют соответствующие компоненты и затем клиппятся общим капом.

### 3.4 Параметры shaping (ручки настройки)

Во всех shaping-окружениях используется один и тот же принцип ограничения влияния shaping на базовую цель:

- `rc_weight`: масштаб shaping-сигнала.
- `max_shaping_ratio`: клиппинг по модулю относительно базовой награды: `|bonus| <= max_shaping_ratio * max(1, |base_reward|)`.
- `rc_pb_gamma`: дисконт PBRS, есть в вариантах `*_pb`, `*_ap`, `*_sp`, `*_asp`.

Замечание про движение: в shaping-классах в `__init__` принудительно выставляется `move_amount=3` (см. `src/envs/starcraft/melee_range_control_*.py`), даже если в YAML стоит другое значение.

---

## 4. Логи, метрики и артефакты (что сохраняется и как читается)

### 4.1 Что именно попадает в лог

- `StarCraft2Env` добавляет в `info` на терминальном шаге эпизодные метрики (например, `shaping/dmin_mean`, `shaping/cooldown`, `shaping/ally_dmg`), которые накапливаются внутри эпизода (см. `src/envs/starcraft/StarCraft2Env.py`).
- shaping-окружения добавляют/обновляют `shaping/*` через `ShapingMetrics` (см. `src/envs/starcraft/utils.py`).

### 4.2 Как runner превращает `info` в графики

`src/runners/parallel_runner.py` и `src/runners/episode_runner.py` собирают `env_info` **на момент завершения эпизода** и логируют среднее по батчу эпизодов как `<key>_mean`.

Практическое следствие: если вы хотите добавить новую метрику в TensorBoard - достаточно вернуть ее ключом в `info` на терминальном шаге.

### 4.3 Где лежат логи

- TensorBoard: `results/tb_logs/<unique_token>/` (создается в `src/run/run.py`).
- Sacred: `results/sacred/<map>/<experiment_name>/...` (создается в `src/main.py`).
- Финальный экспорт: `final_tb_logs/` и `final_metrics/` (см. `plot_final_metrics.py`).

TensorBoard:

```bash
./run_tensorboard.sh
```

Генерация графиков из финальных логов:

```bash
python plot_final_metrics.py
```

---

## 5. Чекпоинты: сохранение, загрузка, оценка

### 5.1 Сохранение моделей

Сохранение включается `save_model=True` (по умолчанию в `src/config/default.yaml` оно выключено).

Логика сохранения - в `src/run/run.py`: каждые `save_model_interval` шагов пишется:

`results/models/<unique_token>/<t_env>/`

### 5.2 Загрузка и оценка

```bash
python src/main.py --config=qmix --env-config=melee_range_control_ab \
  with checkpoint_path="results/models/<unique_token>" load_step=0 evaluate=True
```

- `load_step=0` берет максимальный доступный шаг.
- `evaluate=True` запускает тестовые эпизоды и завершает процесс.

---

## 6. Дипломный протокол одной командой

`run_all_shapings.sh` - оркестратор экспериментов:

- перебирает карты и задает per-map `epsilon_anneal_time`,
- прогоняет все shaping конфиги + baseline `sc2`,
- чистит процессы SC2 между запусками,
- фиксирует seed (`seed=42`) для воспроизводимости,
- включает чекпоинты (`save_model=True`, `save_model_interval=50000`).

Запуск:

```bash
bash run_all_shapings.sh
```

Кастомный список карт (формат `map:epsilon_anneal_time`):

```bash
bash run_all_shapings.sh "3s_vs_3z:180000" "3s_vs_4z:220000"
```

---

## 7. Как добавить новый shaping-вариант (минимальный путь)

1. Создать новый класс окружения в `src/envs/starcraft/` (удобно начать с копии `melee_range_control_sb.py`).
2. Зарегистрировать его в `src/envs/__init__.py`.
3. Добавить YAML в `src/config/envs/<name>.yaml` с `env: <name>` и нужными `env_args`.
4. (Опционально) включить в `run_all_shapings.sh` для массового прогона.

---

## 8. Презентационный пакет для диплома

Каталог `MARL Reward Shaping/` содержит копии файлов, разложенные по структуре "Исходные файлы программного продукта" под требования методички. Он не участвует в импортах/запуске; исполняемая версия проекта - в корне репозитория.

---

## 9. Атрибуция

Проект основан на PyMARL2 и SMAC. Лицензия - см. `LICENSE`.
