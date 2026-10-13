# Module 2 - House Prices: предобработка и регрессия

**Автор:** Дремин Иларион Алексеевич, АСОиУб-23-2  
**Дата:** 2026-10-13

## Результаты

| Модель | RMSE (log) | MAE (log) | R² (log) | RMSE ($) |
|--------|-----------:|----------:|---------:|---------:|
| Linear Regression | 0.1259 | 0.0849 | 0.8879 | $19,640 |
| Ridge | 0.1219 | 0.0825 | 0.8949 | $18,745 |
| **Lasso** | **0.1147** | **0.0771** | **0.9069** | **$17,453** |
| ElasticNet | 0.1149 | 0.0772 | 0.9067 | **$17,408** |

**Лучшая модель по RMSE в log-пространстве:** Lasso  
**Лучшая модель по RMSE в долларах:** ElasticNet

Lasso также выполнил отбор признаков:
- ненулевых признаков: 116
- отброшенных признаков: 156

## 🚀 Быстрый старт

```python
import joblib
import requests
from io import BytesIO

BASE_URL = "https://raw.githubusercontent.com/DreminIlarion/ml-course-dremin/main/module-2-house-prices"

model = joblib.load(
    BytesIO(requests.get(f"{BASE_URL}/models/lasso_model.pkl").content)
)

scaler = joblib.load(
    BytesIO(requests.get(f"{BASE_URL}/models/scaler.pkl").content)
)