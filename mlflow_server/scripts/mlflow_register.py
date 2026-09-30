import json
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import yaml
from dotenv import load_dotenv
from os import getenv


# Параметры .env:
load_dotenv()
EXPERIMENT_NAME = getenv('EXPERIMENT_NAME', 'buildings_flats_price_prediction')
MODEL_NAME = getenv('MODEL_NAME', 'buildings_flats_price_model')
MLFLOW_URI = getenv('MLFLOW_URI', 'http://localhost:5000')



# Концигурация
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Загрузка параметров:
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Загрузка модели
with open('models/fitted_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Загрузка обучающих данных
target_col = params['data']['target_col']
train_df = pd.read_csv('data/train.csv')
X_train = train_df.drop(columns=[target_col])
input_example = X_train.iloc[[0]]
predictions = model.predict(X_train.iloc[:5])
signature = infer_signature(X_train.iloc[:5], predictions)

# Загрузка метрик валидации
with open('cv_results/metrics.json', 'r') as f:
    metrics = json.load(f)

# Запуск MLflow и логирование
with mlflow.start_run(run_name='buildings_flats_final_model') as run:
    # Логируем параметры модели из DVC
    model_params = params['model']
    mlflow.log_param('model_depth', model_params['depth'])
    mlflow.log_param('model_iterations', model_params['iterations'])
    mlflow.log_param('model_learning_rate', model_params['learning_rate'])
    mlflow.log_param('model_loss_function', model_params['loss_function'])
    mlflow.log_param('random_state', model_params['random_state'])

    # Логируем метрики валидации (по одной)
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

    # Логируем обучающие данные как artifact
    mlflow.log_artifact('data/train.csv', artifact_path='training_data')
    mlflow.log_artifact('cv_results/metrics.json', artifact_path='metrics')

    # Регистрируем модель: передаём signature и input_example
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path='model',
        signature=signature,
        input_example=input_example,
        registered_model_name='buildings_flats_price_model',
    )

    # логируем EDA:
    mlflow.log_artifact(
            local_path="model_improvement/EDA.ipynb",
            artifact_path="eda",
        )
    mlflow.log_artifact(
        local_path="model_improvement/EDA_conclusion.md",
        artifact_path="eda",
    )

    print(f'Run ID: {run.info.run_id}')
    print(f'Зарегистрирована модель: {MODEL_NAME}')
