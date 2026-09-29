# Module 1 – Titanic: EDA + бинарная классификация

**Автор:** Дремин Иларион Алексеевич, АСОиУб-23-2  
**Дата:** 2026-09-29

## Описание

Полный цикл ML-проекта на датасете «Титаник»: EDA, обработка пропусков, feature engineering, обучение двух моделей (Logistic Regression, Decision Tree), оценка качества, интерпретация и функция предсказания для нового пассажира.

## Результаты

| **Модель**          | **Accuracy** | **Precision** | **Recall** | **F1** | **ROC-AUC** |
| ------------------- | ------------ | ------------- | ---------- | ------ | ----------- |
| Logistic Regression | 0.8045       | 0.7833        | 0.6812     | 0.7287 | 0.8486      |
| Decision Tree       | 0.7821       | 0.7419        | 0.6667     | 0.7023 | 0.8132      |

**Время обучения:** ~0.079 сек (LR), ~0.018 сек (DT)

## Быстрый старт

```python
import joblib, requests, json
from io import BytesIO

BASE_URL = "https://raw.githubusercontent.com/DreminIlarion/ml-course-dremin/main/module-1-titanic"

model = joblib.load(
    BytesIO(requests.get(f"{BASE_URL}/models/lr_model.pkl").content)
)

scaler = joblib.load(
    BytesIO(requests.get(f"{BASE_URL}/models/scaler.pkl").content)
)

feature_cols = json.loads(
    requests.get(f"{BASE_URL}/models/feature_cols.json").content
)
