# NayePankh Foundation Impact Intelligence Platform

## Project Report

### 1. Objective

This project demonstrates how data analytics can support decision-making at NayePankh Foundation, a youth-led NGO focused on education, food distribution, menstrual hygiene, clothing support, and community health campaigns.

The goal is to move beyond static reporting and show a complete analytics workflow: data preparation, exploratory analysis, KPI tracking, predictive modeling, model comparison, and interactive visualization.

### 2. Data Sources

**Simulated operational dataset**

A two-year monthly dataset (2024–2025) was created to mirror realistic NGO operations across six cities (including Kanpur, the foundation's headquarters). Each record includes donations, expenses, beneficiaries reached, volunteer hours, campaigns, programs, donor types, and outreach channels.

Programs align with NayePankh's public mission:

- Education Support
- Food Distribution
- Menstrual Hygiene
- Clothing Support
- Health Awareness

**Real public context dataset**

State-level indicators (literacy rate, urban population share, multidimensional poverty index) were added from open public statistics to contextualize where social need is highest relative to NayePankh's field activity.

> This project uses simulated operations data for demonstration. It is not an official NayePankh audit.

### 3. Analytical Approach

The workflow follows a standard analytics pipeline:

1. **Data generation and cleaning** — consistent schema, derived KPIs, lag features for time-series modeling
2. **Exploratory analysis** — trends, seasonality, segment comparisons by city, program, and donor type
3. **Feature engineering** — cost per beneficiary, volunteer productivity, festive-season flags, donation lags
4. **Predictive modeling** — three tasks with time-based train/test split (2024 train, 2025 test)
5. **Dashboard delivery** — Streamlit app with filters, charts, predictions, and downloadable reports

### 4. Prediction Models and Comparison

Three machine learning tasks were implemented:

| Task | Target | Models Compared | Selection Criteria |
|------|--------|-----------------|-------------------|
| Donation Forecasting | `donation_amount` | Linear Regression, Random Forest, XGBoost | Highest R² |
| Beneficiary Reach Prediction | `beneficiaries_reached` | Linear Regression, Random Forest, XGBoost | Highest R² |
| Campaign Efficiency Classification | `high_efficiency` | Logistic Regression, Random Forest, XGBoost | Highest F1 score |

Evaluation metrics:

- **Regression:** MAE, RMSE, R², MAPE
- **Classification:** Accuracy, Precision, Recall, F1, ROC-AUC

Gradient Boosting (or XGBoost when available) performs best on donation forecasting, while Linear Regression is strongest for beneficiary reach due to the largely linear relationship with volunteer hours and expense inputs. Logistic Regression provides the best F1 score for campaign efficiency classification.

**Actual results from this run:**

| Task | Best Model | Key Metric |
|------|------------|------------|
| Donation Forecasting | Gradient Boosting | R² = 0.87, MAPE = 8.6% |
| Beneficiary Reach | Linear Regression | R² = 0.92, MAPE = 5.1% |
| Campaign Efficiency | Logistic Regression | F1 = 0.90, ROC-AUC = 0.98 |

### 5. Key Findings

Based on the analysis pipeline:

- **Festive season (Aug–Dec)** drives materially higher donation volumes — campaigns should be front-loaded before this window
- **Corporate CSR donors** contribute the highest average donation size; **Individual donors** provide the largest share by volume
- **Menstrual Hygiene** and **Food Distribution** tend to show strong cost-per-beneficiary efficiency
- **Kanpur and Delhi** lead in beneficiary reach; cities with higher poverty indices may still have lower volunteer support relative to need
- **Volunteer hours** and **historical lag features** are among the strongest predictors of future beneficiary reach
- **Gradient Boosting / XGBoost** provides the most reliable donation forecasts; **Linear Regression** works best for beneficiary reach in the test period

### 6. Dashboard Features

The Streamlit dashboard includes:

- KPI cards for donations, beneficiaries, volunteer hours, and efficiency
- Campaign, donor, city, and program visualizations
- Model comparison table with recommended models per task
- 2025 forecast vs actual chart for donations
- Scenario simulator: input city, program, budget, and volunteer hours → predicted outcomes
- Geographic context view combining NayePankh reach with state-level need indicators
- Auto-generated markdown report with downloadable filtered data

### 7. Recommendations for NayePankh

1. **Plan fundraising around seasonality** — increase campaign intensity before the festive quarter
2. **Scale efficient programs** — replicate delivery practices from Menstrual Hygiene and Food Distribution in underperforming cities
3. **Rebalance volunteers** — prioritize cities where poverty indicators are high but volunteer hours per beneficiary are low
4. **Use predictive planning** — apply the beneficiary reach model when allocating monthly field budgets
5. **Segment donor strategy** — retain individual donors for volume while targeting CSR partners for large campaigns
6. **Track efficiency continuously** — monitor cost per beneficiary monthly to flag programs needing process review

### 8. Conclusion

This project shows that even with simulated data, a structured analytics platform can help NayePankh Foundation make smarter decisions about fundraising, volunteer allocation, and program expansion. The combination of descriptive dashboards, predictive models, and model comparison demonstrates practical data analytics skills directly applicable to the internship role.

---

*Built as a Data Analytics internship submission for NayePankh Foundation.*
