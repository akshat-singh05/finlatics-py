# 📊 Finlatics-Py

> End-to-end data analysis pipeline for processing and visualizing online advertising performance data.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.0+-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🎯 Objective

This project demonstrates a complete **data science workflow** applied to online advertising performance data:

- **Clean & preprocess** raw data with production-grade pipelines
- **Perform EDA** with distribution analysis, correlation mapping, and outlier detection
- **Engineer features** — derive KPIs like CTR, CPC, ROAS, ROI, and conversion rates
- **Build ML models** for predicting campaign performance *(coming soon)*
- **Generate insights** through 8 professional-quality visualizations

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Core language |
| **pandas** | Data manipulation & aggregation |
| **NumPy** | Numerical operations |
| **matplotlib** | Data visualization |
| **seaborn** | Statistical charts |
| **scikit-learn** | Machine learning (classification & regression) |
| **SHAP** | Model explainability (Shapley values) |
| **joblib** | Model serialization |

---

## 📁 Project Structure

```
finlatics-py/
├── analysis/                   # Core analysis modules
│   ├── __init__.py
│   ├── data_loader.py          # CSV loading & schema validation
│   ├── data_cleaning.py        # Missing values, dedup, normalization
│   ├── metrics.py              # KPI computation & aggregation
│   ├── visualizations.py       # 8 professional chart generators
│   └── insights.py             # Business insights & budget recommendations
├── models/                     # Machine learning pipeline
│   ├── __init__.py
│   ├── feature_engineering.py  # Target creation, encoding, scaling, splitting
│   ├── trainer.py              # sklearn Pipeline + RandomizedSearchCV
│   ├── evaluator.py            # Evaluation metrics & comparison charts
│   ├── explainability.py       # SHAP analysis & visualization
│   ├── ml_visualizations.py    # Prediction diagnostics & residuals
│   └── saved/                  # Serialized best pipelines (.joblib)
├── data/
│   └── online_advertising_performance_data.csv
├── scripts/
│   └── generate_data.py        # Synthetic dataset generator
├── output/                     # Generated charts (auto-created)
├── main.py                     # Pipeline entry point (analysis + ML)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/akshat-singh05/finlatics-py.git
cd finlatics-py
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate sample data (optional — data is included)

```bash
python scripts/generate_data.py
```

### 4. Run the full pipeline (analysis + ML)

```bash
python main.py                           # Default: high ROI classification
python main.py --target revenue          # Revenue prediction (regression)
python main.py --target high_conversion  # Conversion classification
python main.py --no-ml                   # Analysis only, skip ML
```

This will:
- Load and inspect the raw dataset
- Clean and preprocess the data
- Compute 7 KPIs (CTR, CPC, CPM, Conversion Rate, ROAS, ROI, Revenue/Conversion)
- Print summary reports to the console
- Save 20 charts to `output/` (analysis + insights + ML + SHAP + diagnostics)
- Train 3 models with **RandomizedSearchCV** hyperparameter tuning (5-fold CV)
- Evaluate, compare, and save the best **sklearn Pipeline**
- Generate **SHAP explainability** report with positive/negative impact analysis
- Produce **executive summary** with budget reallocation recommendations

---

## 📈 Visualizations Generated

| # | Chart | Description |
|---|-------|-------------|
| 1 | Campaign Performance | Spend vs Revenue comparison by campaign |
| 2 | Channel Breakdown | Donut charts for spend & revenue by channel |
| 3 | Spend vs Revenue | Scatter plot with break-even line |
| 4 | CTR Trends | Weekly click-through rate by channel |
| 5 | Correlation Heatmap | Feature correlation matrix |
| 6 | Device Distribution | CTR, conversion rate, ROAS by device |
| 7 | ROI Comparison | ROI by campaign and channel |
| 8 | Monthly Trends | Spend, revenue, and conversions over time |
| **9** | **Model Comparison** | Test accuracy, F1, and CV mean side-by-side |
| **10** | **Feature Importance** | Top 15 features driving the best model |
| **11** | **Confusion Matrices** | Per-model confusion matrices |
| **12** | **CV Score Distribution** | Box plot of cross-validation fold scores |
| **13** | **SHAP Summary** | Beeswarm plot: feature impact direction + magnitude |
| **14** | **SHAP Importance** | Mean |SHAP| bar chart (global importance) |
| **15** | **SHAP Waterfall** | Positive vs negative impact breakdown |
| **16** | **Channel Efficiency** | ROAS with profit/loss annotations |
| **17** | **Campaign Scorecard** | Efficiency ranking + revenue vs spend |
| **18** | **Budget Allocation** | Current vs recommended (ROAS-optimized) |
| **19** | **Classification Analysis** | Correct vs incorrect + model comparison |
| **22** | **Model Diagnostics** | F1/R2 scores with training time |

---

## 📊 Key Metrics Computed

| Metric | Formula |
|--------|---------|
| **CTR** | Clicks ÷ Impressions |
| **CPC** | Spend ÷ Clicks |
| **CPM** | (Spend ÷ Impressions) × 1000 |
| **Conversion Rate** | Conversions ÷ Clicks |
| **ROAS** | Revenue ÷ Spend |
| **ROI** | (Revenue − Spend) ÷ Spend |
| **Rev/Conversion** | Revenue ÷ Conversions |

---

## 🔮 Roadmap

- [x] Data loading & validation
- [x] Data cleaning pipeline
- [x] KPI computation engine
- [x] Professional visualizations (8 analysis charts)
- [x] Advanced feature engineering (derived + interaction + time features)
- [x] sklearn Pipeline architecture (ColumnTransformer + Model)
- [x] Hyperparameter tuning (RandomizedSearchCV)
- [x] 5-fold cross-validation with score distribution
- [x] Overfitting prevention (regularization, depth control, subsampling)
- [x] Best pipeline serialization (joblib)
- [x] SHAP model explainability (summary, importance, waterfall)
- [x] Business insights engine (channel efficiency, budget recommendations)
- [x] ML diagnostic visualizations (prediction analysis, residuals)
- [ ] Interactive dashboard

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/akshat-singh05">Akshat Singh</a>
</p>
