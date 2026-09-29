# Проект "Улучшение baseline-модели"

## О проекте

Репозиторий содержит решение проекта 2 спринта: улучшение baseline модели для прогнозирования стоимости квартир Яндекс Недвижимости.

## Хранилище артефактов

**Имя бакета:** `s3-student-mle-20260908-0538921743`

Endpoint: `https://storage.yandexcloud.net`

Чтобы проверить артефакты, выполните `dvc pull` в директории `part2_dvc/` —
будут скачаны датасеты (`data/*.csv`) и обученная модель
(`models/fitted_model.pkl`).

---

## Этап 1. Разворачивание MLflow Tracking Server и MLflow Model Registry. Регистрация существующей модели.

### 1.1. Настройка окружения

Для работы используются два независимых `.env`-файла:

1. **`.env` для MLflow-сервера** — инфраструктурные параметры (БД, S3).
2. **`.env` для регистрации модели** — параметры подключения к MLflow и имена сущностей.

#### `.env` для MLflow-сервера

Скопируйте шаблон переменных окружения и заполните его своими значениями:

```bash
cp mlflow_server/mlflow_env_template mlflow_server/.env
```

В `.env` должны быть заданы следующие переменные:

| Переменная | Назначение |
|---|---|
| `MLFLOW_S3_ENDPOINT_URL` | Адрес S3-совместимого хранилища для артефактов |
| `AWS_ACCESS_KEY_ID` | Ключ доступа к S3 |
| `AWS_SECRET_ACCESS_KEY` | Секретный ключ S3 |
| `S3_BUCKET_NAME` | Имя бакета для артефактов |
| `DB_DESTINATION_USER` | Пользователь БД (backend store) |
| `DB_DESTINATION_PASSWORD` | Пароль БД |
| `DB_DESTINATION_HOST` | Хост БД |
| `DB_DESTINATION_PORT` | Порт БД |
| `DB_DESTINATION_NAME` | Имя базы данных |
| `EXPERIMENT_NAME` | Имя эксперимента в MLflow |
| `MODEL_NAME` | Имя регистрируемой модели |
| `MLFLOW_URI` | URI MLflow-сервера (по умолчанию `http://localhost:5000`) |

#### `.env` для регистрации модели

```bash
cp mlflow_server/registrate_env_template .env
```

В этом `.env` указываются только параметры, необходимые скрипту `mlflow_register.py`:

| Переменная | Обязательность | Значение по умолчанию | Назначение |
|---|---|---|---|
| `EXPERIMENT_NAME` | опционально | `buildings_flats_price_prediction` | Имя эксперимента в MLflow, в который логируется run |
| `MODEL_NAME` | опционально | `buildings_flats_price_model` | Имя регистрируемой модели в MLflow Model Registry |
| `MLFLOW_URI` | опционально | `http://localhost:5000` | URI MLflow-сервера (tracking URI) |

> Переменные `EXPERIMENT_NAME`, `MODEL_NAME` и `MLFLOW_URI` имеют значения по умолчанию в коде, поэтому `.env` для регистрации модели можно не создавать, если используются дефолтные значения. Однако при работе с удалённым MLflow или другим именем эксперимента/модели — файл обязателен.

#### Зависимости

Установите зависимости:

```bash
pip install -r requirements.txt
```

### 1.2. Запуск MLflow-сервера

Скрипт для поднятия MLflow-сервисов: [`mlflow_server/run_mlflow_server.sh`](mlflow_server/run_mlflow_server.sh)

Скрипт выполняет:
- загрузку переменных из `.env`;
- экспорт переменных для доступа к S3;
- запуск `mlflow server` с backend-store в PostgreSQL и artifact-root в S3.

Запуск:

```bash
cd mlflow_server
./run_mlflow_server.sh
```

После запуска MLflow UI доступен по адресу `http://localhost:5000`.

### 1.3. Регистрация модели:

Скрипт регистрации модели: [`mlflow_server/scripts/mlflow_register.py`](mlflow_server/scripts/mlflow_register.py)

Скрипт выполняет:
- загрузку `.env` (параметры подключения к MLflow);
- подключение к MLflow и установку эксперимента;
- загрузку параметров из `params.yaml`;
- загрузку обученной модели из `models/fitted_model.pkl`;
- загрузку обучающих данных и метрик валидации;
- логирование параметров, метрик и артефактов;
- регистрацию модели под именем `MODEL_NAME`.

Запуск:

```bash
python mlflow_server/scripts/mlflow_register.py
```

После успешного выполнения в консоль выводятся `Run ID` и имя зарегистрированной модели.

### 1.4. Эксперимент в MLflow

- **ID эксперимента:** `623d2020643346999239cdd6a8ae15af`
- **Название эксперимента:** `buildings_flats_price_prediction`

Ссылка на эксперимент в MLflow UI:

```
http://localhost:5000/#/experiments/2/runs/623d2020643346999239cdd6a8ae15af
```

![](docs/imgs/first_experiment.png)