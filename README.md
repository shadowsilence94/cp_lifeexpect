# 🌍 Life Expectancy Prediction using Machine Learning

An interactive web application showcasing machine learning techniques for predicting and forecasting life expectancy across countries using socioeconomic and health indicators.

## 👥 Team Members
- **Htut Ko Ko**
- **Kaung Hein Htet**
- **Michael R. Lacar**

**Course:** AT82.01 – Computer Programming for Data Science and AI

## 🎯 Project Overview

This project demonstrates advanced machine learning techniques to:
- Predict life expectancy based on multiple socioeconomic factors
- Forecast future life expectancy trends (2025-2030)
- Visualize global health trends and patterns
- Compare model performances and analyze feature importance

## ✨ Key Features

- **🤖 Multiple ML Models:** Ridge Regression, Random Forest, XGBoost with hyperparameter tuning
- **📊 Comprehensive Data Preprocessing:** Missing value imputation, feature engineering, temporal features
- **⏰ Time Series Validation:** Prevents data leakage using temporal train/test splits
- **🔮 Future Forecasting:** Predictions for 2025-2030 based on projected trends
- **📈 Interactive Visualizations:** Built with Plotly and Streamlit
- **🌐 Global Analysis:** Compare countries, regions, and track improvements over time

## 📚 Dataset

The project uses World Bank Development Indicators covering:
- **Countries:** 200+
- **Time Period:** 1975-2024
- **Total Records:** 10,000+
- **Features:** 17 socioeconomic and health indicators

### Key Features Used:
- GDP per Capita (USD)
- Health Expenditure (% GDP & per Capita)
- Infant Mortality Rate
- Access to Clean Fuels
- PM 2.5 Air Pollution
- Available Physicians
- Income Distribution
- Fertility Rate
- Age Dependency Ratio
- Education Expenditure

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd cp_life
```

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

3. **Run the Jupyter notebook first (if not already done):**
```bash
jupyter notebook Life_Expectancy_ML_Pipeline.ipynb
```
Execute all cells to generate the required data files.

4. **Launch the web application:**
```bash
streamlit run app.py
```

5. **Open your browser:**
The app will automatically open at `http://localhost:8501`

## 📱 Application Features

### 🏠 Overview Page
- Project objectives and methodology
- Dataset statistics
- Global life expectancy trends

### 🔮 Forecast Page
- Country-specific forecasting (2025-2030)
- Interactive predictions with confidence intervals
- Top/Bottom performers comparison
- Forecast statistics and growth rates

### 📈 ML Pipeline Page
- Data processing workflow visualization
- Data quality analysis
- Feature engineering explanation
- Temporal train/validation/test split

### 🎯 Model Performance Page
- Model comparison (Ridge, Random Forest, XGBoost)
- R², RMSE, MAE metrics
- Hyperparameter tuning results
- Cross-validation performance

### 📊 Feature Analysis Page
- Feature importance rankings
- Correlation analysis
- Scatter plots showing key relationships
- GDP and infant mortality impact visualization

### 🌐 Global Trends Page
- Historical trends (1975-2024)
- Top performing countries
- Greatest improvements over 20 years
- Global statistics summary

## 🧠 Machine Learning Pipeline

### 1. Data Preprocessing
- **Missing Value Handling:** Forward fill, interpolation, and median imputation
- **Feature Engineering:** Temporal lag features, 3-year moving averages
- **Scaling:** StandardScaler for feature normalization

### 2. Model Training
- **Time Series Split:** Train (1975-2017), Validation (2018-2020), Test (2021-2024)
- **Cross-Validation:** Time Series Cross-Validation with 5 folds
- **Hyperparameter Tuning:** Grid Search CV

### 3. Models Evaluated
| Model | Test R² | Test RMSE | Test MAE |
|-------|---------|-----------|----------|
| Ridge Regression | 0.937 | 2.18 | 1.72 |
| **Random Forest** | **0.959** | **1.76** | **1.28** |
| XGBoost | 0.958 | 1.78 | 1.30 |

**Best Model:** Random Forest with 95.9% variance explained

### 4. Forecasting
- Projects key indicators based on historical trends
- Conservative growth assumptions
- GDP growth rates vary by income level
- Environmental improvements factored in

## 📊 Key Results

- **Model Accuracy:** 95.9% R² score (Random Forest)
- **Prediction Error:** ±1.76 years RMSE
- **Global Trend:** Life expectancy increased from ~59 years (1975) to ~73 years (2024)
- **Forecast:** Continued improvement projected through 2030

## 🛠️ Technologies Used

- **Python 3.8+**
- **Streamlit:** Interactive web application
- **Plotly:** Interactive visualizations
- **Pandas & NumPy:** Data manipulation
- **Scikit-learn:** Machine learning models
- **XGBoost:** Gradient boosting
- **Jupyter:** Notebook development

## 📁 Project Structure

```
cp_life/
│
├── app.py                              # Streamlit web application
├── Life_Expectancy_ML_Pipeline.ipynb  # ML pipeline notebook
├── Data_Preporcessing.ipynb           # Data preprocessing notebook
├── requirements.txt                    # Python dependencies
├── README.md                          # Project documentation
│
├── data/
│   ├── world_bank_data_cleaned.csv          # Cleaned dataset
│   ├── world_bank_data_imputed.csv          # Imputed dataset
│   └── life_expectancy_forecasts_2025_2030.csv  # Forecast results
│
└── heatmap.png                        # Correlation heatmap
```

## 🎓 Learning Outcomes

This project demonstrates proficiency in:
- **Data Science:** Data cleaning, EDA, feature engineering
- **Machine Learning:** Model selection, hyperparameter tuning, evaluation
- **Time Series Analysis:** Temporal validation, forecasting
- **Data Visualization:** Interactive dashboards, storytelling with data
- **Software Engineering:** Clean code, documentation, reproducibility

## �� Usage Examples

### View Country-Specific Forecast:
1. Navigate to "🔮 Forecast" page
2. Select a country from the dropdown
3. View historical trends and future predictions
4. Analyze forecast statistics

### Compare Model Performance:
1. Go to "🎯 Model Performance" page
2. Review R² and RMSE comparisons
3. Examine cross-validation results
4. Understand hyperparameter tuning

### Analyze Feature Importance:
1. Open "📊 Feature Analysis" page
2. View feature importance rankings
3. Explore correlation heatmap
4. Examine scatter plots for key relationships

## 🤝 Contributing

This is an academic project for demonstration purposes. Feedback and suggestions are welcome!

## 📝 License

This project is developed for educational purposes as part of the AT82.01 course.

## 📧 Contact

For questions or feedback, please contact:
- Htut Ko Ko
- Kaung Hein Htet
- Michael R. Lacar

## 🙏 Acknowledgments

- **Data Source:** World Bank Development Indicators
- **Course:** AT82.01 – Computer Programming for Data Science and AI
- **Instructors:** [Course Instructors]

---

**Note:** This project is for educational and research purposes. Predictions should be interpreted with appropriate caution and not used for policy decisions without further validation.
