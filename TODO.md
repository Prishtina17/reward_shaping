# TODO: финальный прогон и публикационная дорожная карта

Этот файл относится только к существующему дипломному проекту про интеграцию
state-based, action-based и potential-based reward shaping в PyMARL2/QMIX.
Цель — превратить проект из одно-seed сравнения вариантов награды в аккуратное
исследование sample efficiency, поведения агентов и границ применимости shaping.

## P0 — заморозить честный финальный протокол

- [x] Уравнять `move_amount=2` у baseline и всех shaping-окружений. Старые
  результаты с `baseline=2`, `shaping=3` считать exploratory: этот конфаунд не
  позволяет приписывать разницу только reward shaping.
- [x] Сделать фиксированный step budget основным протоколом. Не останавливать
  разные методы в разные моменты по шумному train win rate.
- [x] Считать steps-to-threshold постфактум по greedy test win rate. Основные
  пороги: 50%, 80%, 95%; засчитывать только устойчивое достижение в трех
  последовательных evaluation-точках и отдельно указывать долю seeds, которые
  вообще достигли порога.
- [x] Добавить пять независимых seeds: `42,43,44,45,46`. Три seeds допустимы
  только для пилота; для финальных выводов целиться минимум в пять.
- [x] Добавить seed в идентификатор TensorBoard/checkpoint и агрегировать все
  seeds, а не выбирать последний запуск.
- [x] Добавить manifest и resume в `run_all_shapings.sh`, чтобы завершенные
  комбинации не пересчитывались после сбоя или перезагрузки.
- [x] Сохранять checkpoint в конце каждого обучения, даже если интервал
  checkpoint не совпал с последним `t_env`.
- [x] Исключить `6h_vs_8z` из обязательного домашнего финального прогона. Карта
  дорогая и меняет исследуемый matchup; ее отсутствие нужно явно указать как
  compute-driven scope decision, а не скрывать.
- [x] Оставить четыре основные карты: `2m_vs_1z`, `3s_vs_3z`, `3s_vs_4z`,
  `3s_vs_5z`.
- [x] Исправить action advice: при слишком малой дистанции — отходить, в зоне
  стрельбы — атаковать при готовом оружии/отходить на cooldown, за пределами
  дальности — сближаться.
- [x] Исправить `rc_melee_only`: Zealot/Zergling/Baneling распознаются как
  melee, остальные поддерживаемые типы не считаются melee автоматически.
- [x] Исправить чистый PBRS: использовать потенциал начального состояния,
  нулевой терминальный потенциал (включая timeout) и не клиппировать компонент
  по умолчанию. Нелинейный clip разрушает телескопическую policy-invariance.
- [x] Исправить задержку state-бонуса на один transition в ASP.
- [x] Выполнить короткий SC2 smoke на baseline и на каждом из семи shaping env.
- [ ] Выполнить пилот в отдельном `RESULTS_ROOT` на 1–2 seeds; проверить
  длительность, место на диске, монотонность `t_env`, наличие всех test-метрик и
  финальных checkpoint. Не смешивать пилот с финальными данными.
- [ ] До финального запуска письменно заморозить commit SHA, карты, seeds,
  `t_max`, test interval, epsilon schedule, threshold rules и primary metric.
  После просмотра финальных результатов гиперпараметры не менять.
- [ ] Записать версии Python, PyTorch, CUDA/CPU, StarCraft II `4.10`, SMAC commit,
  ОС и железо. Время wall-clock можно дать как практический контекст, но основная
  ось sample efficiency — environment steps.
- [ ] Запустить полную матрицу. Для каждого map/env требовать ровно пять
  завершенных seeds в manifest и TensorBoard.
- [ ] Сохранить исходные event-файлы, Sacred configs, aggregate CSV и команду
  запуска. Графики без raw logs недостаточны для воспроизводимости.

### Если compute все равно не хватает

1. Не уменьшать число seeds первым шагом.
2. Сначала оставить baseline, чистый PBRS, лучший non-potential вариант и лучший
   комбинированный вариант; остальные комбинации обозначить как расширенную
   ablation из диплома.
3. Затем уменьшить единый `t_max` на основании отдельного пилота, одинаково для
   всех методов.
4. Только в крайнем случае публиковать 3 seeds как preliminary result и прямо
   отмечать ограничение. Evaluation episodes не заменяют независимые seeds.

## P1 — исследовательский вопрос вместо «сравнили семь шейпингов»

### Основная подача

Рабочая формулировка:

> Как state-, action- и potential-based reward shaping влияют не только на
> итоговый win rate QMIX, но и на sample efficiency и механизм формирования
> kiting-поведения в cooperative partially observable MARL?

Подходящие research questions:

- **RQ1 — sample efficiency:** сколько environment steps требуется до заранее
  заданного устойчивого win rate?
- **RQ2 — конечное качество:** сохраняется ли преимущество при одинаковом
  фиксированном бюджете или shaping только ускоряет раннее обучение?
- **RQ3 — механизм:** связаны ли изменения win rate с ожидаемым поведением —
  дистанцией до ближайшего противника, использованием cooldown, входом в sweet
  zone, входящим/исходящим damage и временем первого убийства?
- **RQ4 — цена эвристики:** дают ли action/state bonuses больший ранний выигрыш,
  но более высокий разброс или худшее конечное поведение, чем строгий PBRS?
- **RQ5 — композиция:** дают ли комбинации компоненты сверх лучшего одиночного
  shaping или эффект объясняется одним доминирующим сигналом?

Это контролируемое causal comparison: меняется только reward signal, а карта,
QMIX, наблюдения, movement, budget, seed set и evaluation protocol совпадают.

### Что считать результатом

- Primary task metric: `test_battle_won_mean` по исходной задаче, не shaped
  return.
- Primary efficiency metric: steps до заранее выбранного устойчивого порога.
- Обязательные дополнительные метрики: normalized AUC, final win rate при общем
  budget, mean ± 95% CI по seeds, reached/not-reached для threshold.
- Behavioral diagnostics: `dmin_mean`, cooldown/weapon idling, ally/enemy alive,
  damage, first allied/enemy kill. Они нужны для объяснения *почему* метод
  изменил обучение, а не как декоративные графики.
- Не выдавать eval episodes за независимые наблюдения: uncertainty считать по
  training seeds.
- Не ранжировать методы по одной лучшей точке или одному seed.
- Сохранить отрицательные результаты. «PBRS не ускорил QMIX после устранения
  конфаунда» или «комбинации не лучше одиночного сигнала» — научно полезный
  вывод при корректном протоколе.

### Теоретические границы claims

- Для чистого PBRS явно выписать
  `F(s_t,s_{t+1}) = gamma * Phi(s_{t+1}) - Phi(s_t)` и условия: одинаковый
  `gamma`, корректный `Phi(s_0)`, нулевой terminal potential.
- `rc_weight * F` остается PBRS с масштабированным потенциалом.
- State/action bonuses и их комбинации могут менять оптимальную политику; не
  называть их policy-invariant.
- Комбинации AP/SP/ASP с эвристическими компонентами и общим clipping также не
  получают гарантию PBRS целиком.
- Теорема PBRS не обещает одинаковую finite-budget динамику deep QMIX с function
  approximation. Именно это позволяет изучать sample efficiency эмпирически.
- Не заявлять SOTA: вклад — контролируемая интеграция, воспроизводимый протокол и
  анализ поведения/эффективности.

## P2 — как усилить актуальность без полного переделывания диплома

### Минимальное внешнее подтверждение на SMACv2

- [ ] Сохранить SMAC1 как основной benchmark для воспроизводимости диплома.
- [ ] Добавить отдельный registry key `sc2v2`, не заменяя `sc2`: старые SMAC
  maps не совместимы со SMACv2 напрямую.
- [ ] Подключить официальный `StarCraftCapabilityEnvWrapper` и
  `32x32_flat.SC2Map`. QMIX/runner/buffer в основном могут остаться прежними,
  потому что SMACv2 сохраняет базовый MultiAgentEnv API; размеры брать только из
  `get_env_info()`.
- [ ] Для дешевого external-validity эксперимента использовать один сценарий
  уровня `protoss_5_vs_5` и три метода: baseline, чистый PBRS, лучший
  heuristic/combined shaping.
- [ ] Проверять новые start positions и unit compositions. Это превращает
  работу из сравнения на фиксированных картах в вопрос о переносимости shaping
  при процедурной вариативности.
- [ ] Перед SMACv2 вынести shaping в композиционную обертку над base env вместо
  семи наследников старого `StarCraft2Env`.
- [ ] Сделать потенциал unit-aware: использовать фактические типы и attack/sight
  ranges. Предположение «все враги melee» и фиксированное кольцо 3–6 нельзя
  переносить на случайные команды SMACv2 без проверки.
- [ ] Не смешивать SMAC1 и SMACv2 в одну среднюю цифру; показывать SMACv2 как
  отдельную проверку robustness/generalization.

SMACv2 — сильное, но необязательное расширение. Сначала нужно получить чистый
multi-seed результат на исправленном SMAC1. Один качественный SMACv2 experiment
полезнее, чем повтор всех восьми вариантов на всех больших сценариях.

## P3 — инженерная подготовка публикационного артефакта

- [ ] После заморозки финальных результатов сократить дублирование семи env:
  общий `RewardShapingWrapper`/набор компонентов AB, SB, PBRS. Не делать большой
  refactor между пилотом и финальным прогоном.
- [ ] Добавить unit tests для ring/potential, terminal transition, action advice,
  melee taxonomy, config parity и seeded aggregation.
- [ ] Добавить одну команду protocol validation: проверить одинаковые базовые
  env args, gamma, budget, интервалы и наличие пяти seeds до анализа.
- [ ] Зафиксировать зависимости lock/constraints-файлом; сохранить pinned SMAC
  commit и точную версию SC2.
- [ ] Убрать из публикационного Git-артефакта видео, слайды, generated images и
  прочие файлы, не нужные для воспроизведения экспериментов. Дипломный пакет
  можно оставить отдельным release/archive.
- [ ] Добавить таблицу «config → математическая добавка → нарушает/сохраняет
  policy invariance → ожидаемый behavioral effect».
- [ ] Опубликовать команды для одного smoke-run, одного seeded-run, полного
  протокола, resume, анализа и evaluation checkpoint.
- [ ] Для каждого рисунка указывать n seeds, полосу uncertainty, fixed budget и
  правило smoothing/interpolation.

## P4 — как написать и подать работу

Варианты названия:

1. **Beyond Win Rate: Reward Shaping, Sample Efficiency and Kiting Behavior in
   Cooperative Multi-Agent Reinforcement Learning**
2. **A Controlled Study of State-, Action- and Potential-Based Reward Shaping
   for QMIX**
3. **What Does Reward Shaping Teach? An Empirical Study of Cooperative
   Micromanagement in SMAC**

Заявляемые contributions без преувеличения:

1. единая реализация трех семейств shaping и их ablations в одном QMIX-контуре;
2. fairness-controlled multi-seed protocol с одинаковой динамикой среды;
3. одновременная оценка sample efficiency, final performance и behavioral
   mechanism;
4. открытый воспроизводимый артефакт; опционально — external-validity check на
   SMACv2.

Предлагаемая структура статьи:

1. проблема sparse/delayed feedback и credit assignment в cooperative MARL;
2. distinction между heuristic shaping и policy-invariant PBRS;
3. реализация и заранее определенные hypotheses;
4. experimental protocol и fairness controls;
5. learning curves + threshold/AUC/final metrics;
6. behavioral diagnostics и разбор failure cases;
7. limitations: один алгоритм, малые SMAC1 maps, home compute, ручной potential;
8. SMACv2/другие алгоритмы как external validation или future work.

Позиционирование:

- PyMARL2 и SMAC1 — инфраструктура, а не novelty. Популярность/stars репозитория
  не доказывают научную значимость; важны воспроизводимость и корректная ссылка
  на upstream.
- Не продавать статью как «новый reward shaping algorithm», если формула не
  новая. Продавать как controlled empirical study и анализ trade-offs.
- Для preprint, студенческой/региональной конференции, workshop или applied
  journal такой scope реалистичен. Для сильного MARL venue желательно добавить
  SMACv2 generalization или второй дешевый алгоритм (например, VDN) только после
  появления устойчивого основного результата.

## Полезные источники для related work

- PyMARL2: <https://github.com/hijkzzz/pymarl2>
- Original SMAC: <https://github.com/oxwhirl/smac>
- SMACv2 code: <https://github.com/oxwhirl/smacv2>
- SMACv2 paper (NeurIPS 2023 Datasets and Benchmarks):
  <https://papers.nips.cc/paper_files/paper/2023/hash/764c18ad230f9e7bf6a77ffc2312c55e-Abstract-Datasets_and_Benchmarks.html>
- PyMARL2 fork used for SMACv2 experiments:
  <https://github.com/benellis3/pymarl2/tree/smacv2-feature-inferrability>
