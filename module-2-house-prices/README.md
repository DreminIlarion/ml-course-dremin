# Module 2 — House Prices: предобработка данных и регрессия

**Автор:** Дремин Иларион Алексеевич  
**Группа:** АСОиУб-23-2  
**Преподаватель:** Игорь Спирин  
**Репозиторий:** https://github.com/DreminIlarion/ml-course-dremin

## Описание

Практический проект по предобработке данных и базовым алгоритмам регрессии на датасете Kaggle House Prices.

В notebook реализованы:
- EDA и минимум 5 визуализаций;
- анализ и обработка пропусков;
- обработка выбросов;
- `log1p(SalePrice)`;
- Feature Engineering;
- ordinal и one-hot encoding;
- StandardScaler;
- Linear Regression, Ridge, Lasso, ElasticNet;
- GridSearchCV;
- RMSE, MAE, R² в логарифмическом и исходном масштабе;
- анализ коэффициентов и остатков;
- `predict_price()` с 5 демонстрационными примерами;
- сохранение моделей через joblib;
- проверка загрузки из raw GitHub.

## Данные

Скачайте `train.csv` и `test.csv` из Kaggle House Prices и положите в `data/`.

## Структура

```text
module-2-house-prices/
├── README.md
├── notebook.ipynb
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── house_prices_info.md
├── models/
│   ├── lr_model.pkl
│   ├── ridge_model.pkl
│   ├── lasso_model.pkl
│   ├── enet_model.pkl
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── feature_cols.json
│   └── metrics.json
├── examples/
│   ├── eda_plots.png
│   ├── residuals.png
│   └── predictions_scatter.png
└── requirements.txt
```

## Запуск

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

В Google Colab можно открыть notebook, предварительно загрузив `data/train.csv` и `data/test.csv`.

