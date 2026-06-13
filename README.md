# NayePankh Foundation Impact Intelligence Platform

Predictive analytics and decision dashboard for a **Data Analytics internship** submission at NayePankh Foundation.

The project demonstrates how an NGO can monitor fundraising, measure social impact, forecast outcomes, and compare machine learning models using a realistic operational workflow.

> **Data note:** Operational records are simulated based on NayePankh's public mission (education, food distribution, menstrual hygiene, clothing, health awareness). State-level context indicators are from open public sources. This is not official NayePankh Foundation data.

## Project Highlights

- Interactive Streamlit dashboard with KPI tracking and filters
- Exploratory analysis notebook with charts and insights
- Three ML tasks with model comparison:
  - Donation forecasting (regression)
  - Beneficiary reach prediction (regression)
  - Campaign efficiency classification (classification)
- Models compared: Linear/Logistic Regression, Random Forest, XGBoost (falls back to scikit-learn Gradient Boosting if XGBoost is unavailable)
- Scenario simulator for planning volunteer hours and budgets
- Geographic context view using real state-level indicators
- Auto-generated downloadable report

## Tools Used

- Python, Pandas, NumPy
- Streamlit, Plotly
- scikit-learn, XGBoost
- Jupyter Notebook

## Project Structure

```text
impact-analytics-dashboard/
├── app.py
├── data/
│   ├── naye_pankh_sample_impact_data.csv
│   ├── india_state_context.csv
│   └── modeling_dataset.csv
├── models/
│   ├── model_comparison.py
│   └── saved/
├── notebooks/
│   └── 01_impact_eda_and_modeling.ipynb
├── outputs/
│   ├── model_metrics.json
│   ├── forecast_vs_actual.csv
│   └── feature importance CSVs
├── report/
│   └── impact_summary_report.md
├── scripts/
│   ├── generate_sample_data.py
│   ├── prepare_data.py
│   └── train_models.py
├── requirements.txt
└── README.md
```

## Setup and Run

Install dependencies:

```bash
cd impact-analytics-dashboard
pip install -r requirements.txt
```

Generate data, prepare features, and train models:

```bash
python scripts/generate_sample_data.py
python scripts/prepare_data.py
python scripts/train_models.py
```

Launch the dashboard:

```bash
streamlit run app.py
```

Open the analysis notebook:

```bash
jupyter notebook notebooks/01_impact_eda_and_modeling.ipynb
```

## Dashboard Sections

| Tab | Purpose |
|-----|---------|
| Overview | KPI cards, donation vs expense trends, program positioning |
| Donations | Campaign and donor-type performance |
| Impact | City-wise reach, social reach, cost efficiency |
| Volunteer Ops | Volunteer hours and productivity rankings |
| Predictions | Model comparison, forecast chart, scenario simulator |
| Geographic Context | State need indicators vs NayePankh field activity |
| Auto Report | Downloadable summary report and filtered dataset |

## Key Metrics

- Total donations and expenses
- Beneficiaries reached
- Volunteer hours
- Cost per beneficiary
- Beneficiaries per volunteer hour
- Model performance: MAE, RMSE, R², F1, ROC-AUC

## Why This Project Matters for NayePankh

NGOs operate with limited resources. This platform helps answer:

- Which campaigns and programs create the strongest impact per rupee?
- Where should volunteers be allocated next month?
- Can donations and beneficiary reach be forecast reliably?
- Which cities combine high social need with under-resourced operations?

## Submission

Created as a practical Data Analytics internship task for NayePankh Foundation. Demonstrates end-to-end analytics: data preparation, EDA, KPI design, visualization, predictive modeling, and actionable recommendations.
