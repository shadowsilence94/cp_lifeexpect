# 📖 Usage Guide - Life Expectancy ML Web Application

## Quick Start

### Option 1: Using the Shell Script (Recommended)
```bash
./run_app.sh
```

### Option 2: Direct Command
```bash
streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`

## 🎯 Navigation Guide

### 📱 Application Pages

#### 1. 🏠 Overview Page
**What you'll see:**
- Project objectives and key features
- Dataset statistics (countries, years, records)
- List of all 11 features used
- Global life expectancy statistics
- Historical trend visualization (1975-2024)

**Use this page to:**
- Understand the project scope
- Get quick statistics about the dataset
- View overall global trends

---

#### 2. 🔮 Forecast Page
**What you'll see:**
- Country selection dropdown
- Interactive forecast visualization (2015-2030)
- Historical data (blue line) vs Forecast (orange dashed line)
- Confidence intervals (shaded area)
- Forecast statistics panel
- Top 10 and Bottom 10 countries comparison for 2030

**How to use:**
1. Select a country from the dropdown menu
2. View the combined historical and forecast chart
3. Check the statistics panel for:
   - Current life expectancy (2024)
   - Predicted life expectancy (2030)
   - Average annual growth rate
4. Scroll down to compare global forecasts

**Example questions you can answer:**
- "What will USA's life expectancy be in 2030?"
- "Which countries are predicted to have the highest life expectancy?"
- "How much improvement is expected for my country?"

---

#### 3. 📈 ML Pipeline Page
**What you'll see:**
- 3-step data processing workflow
- Missing values analysis
- Data availability by decade
- Feature engineering explanation
- Train/Validation/Test split timeline

**Use this page to:**
- Understand how the data was processed
- Learn about temporal feature engineering
- See why time-series splitting prevents data leakage
- Understand the model training workflow

**Key concepts explained:**
- Lag features (t-1)
- Moving averages (3-year)
- Temporal validation

---

#### 4. 🎯 Model Performance Page
**What you'll see:**
- Side-by-side model comparison charts
- R² scores (higher is better)
- RMSE scores (lower is better)
- Best model metrics
- Hyperparameter tuning results
- Cross-validation performance

**Use this page to:**
- Compare different ML algorithms
- Understand model accuracy
- See which model was selected and why
- Learn about hyperparameter optimization

**Key metrics:**
- **R² = 0.959**: Model explains 95.9% of variance
- **RMSE = 1.76**: Average error is ±1.76 years
- **MAE = 1.28**: Typical error is 1.28 years

---

#### 5. 📊 Feature Analysis Page
**What you'll see:**
- Feature importance bar chart
- Correlation analysis
- Scatter plots showing relationships:
  - GDP per Capita vs Life Expectancy
  - Infant Mortality vs Life Expectancy

**Use this page to:**
- Understand which factors matter most
- See how economic/health indicators correlate
- Validate the model's decision-making process

**Top 3 most important features:**
1. **Infant Mortality** (28.5%)
2. **GDP per Capita** (21.5%)
3. **Health Expenditure per Capita** (16.5%)

---

#### 6. 🌐 Global Trends Page
**What you'll see:**
- Historical range chart (max, min, average)
- Top 10 performing countries table and trends
- Greatest improvements analysis (last 20 years)
- Global statistics summary

**Use this page to:**
- Compare countries over time
- Identify success stories
- See the global gap between highest and lowest
- Track improvements by country

**Example insights:**
- "Which countries improved the most?"
- "Is the global gap narrowing?"
- "What are the long-term trends?"

---

## 🔧 Interactive Features

### Hover Interactions
- **Charts:** Hover over any data point to see exact values
- **Tooltips:** Hover over metric cards for additional information

### Zoom and Pan
- **Plotly charts support:**
  - Click and drag to zoom
  - Double-click to reset view
  - Use scroll wheel to zoom
  - Pan by clicking and dragging

### Country Selection
- **Forecast page:** Type to search for countries
- **Auto-complete:** Start typing to filter options

---

## 💡 Tips for Best Experience

### 1. Understanding the Visualizations

**Line Charts:**
- Blue = Historical data
- Orange = Forecasts
- Shaded areas = Confidence intervals

**Bar Charts:**
- Green colors = Good performance
- Red colors = Lower performance
- Compare heights for relative values

**Scatter Plots:**
- Each point = One country-year observation
- Trend lines show overall relationship
- Look for patterns and outliers

### 2. Interpreting Forecasts

**What forecasts show:**
- Projected trends based on current trajectories
- Realistic growth assumptions
- Country-specific predictions

**What forecasts DON'T account for:**
- Major disruptions (pandemics, wars)
- Breakthrough medical technologies
- Sudden policy changes

**Confidence intervals indicate:**
- Wider bands = More uncertainty
- Narrower bands = More confident predictions

### 3. Model Performance Metrics

**R² Score (0-1):**
- 0.959 means 95.9% of variance explained
- Higher is better
- 0.95+ is excellent for life expectancy

**RMSE (Root Mean Square Error):**
- Average prediction error in years
- 1.76 years is very good
- Lower is better

**MAE (Mean Absolute Error):**
- Typical error magnitude
- 1.28 years means most predictions within ±1.3 years
- Lower is better

---

## 🎓 Educational Use Cases

### For Students:
1. **Understand ML Pipeline:** Follow the complete workflow from data to predictions
2. **Learn Feature Engineering:** See how temporal features improve predictions
3. **Model Comparison:** Understand why Random Forest outperformed Ridge Regression
4. **Visualization Skills:** Learn how to present ML results effectively

### For Presentations:
1. **Start with Overview:** Set context and objectives
2. **Show Forecast:** Demonstrate practical application
3. **Explain Pipeline:** Show technical depth
4. **Present Results:** Model performance and validation
5. **Insights:** Feature importance and global trends

### For Research:
1. **Methodology:** ML Pipeline page shows complete approach
2. **Results:** Model Performance page has all metrics
3. **Validation:** Cross-validation and temporal splitting
4. **Feature Analysis:** Understand driver variables

---

## 🐛 Troubleshooting

### App won't start
```bash
# Check if Streamlit is installed
pip install streamlit plotly

# Try running with full path
python -m streamlit run app.py
```

### Data files not found
```bash
# Verify data files exist
ls data/*.csv

# If missing, run the Jupyter notebook first
jupyter notebook Life_Expectancy_ML_Pipeline.ipynb
```

### Port already in use
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

### Slow loading
- Data is cached after first load
- Sample data is used for performance
- Wait for initial cache building

---

## 📊 Data Sources

All data comes from:
- **World Bank Development Indicators**
- Covering 200+ countries
- Time period: 1975-2024
- Updated regularly

---

## 🤔 FAQ

**Q: How accurate are the forecasts?**
A: The model has 95.9% R² on test data, with ±1.76 year average error. Forecasts use conservative growth assumptions.

**Q: Can I download the data?**
A: Yes, the CSV files are in the `data/` directory.

**Q: Why use temporal splitting?**
A: To prevent data leakage. The model only learns from past data to predict future, like real forecasting.

**Q: Which countries are included?**
A: 200+ countries with World Bank data availability.

**Q: How often is data updated?**
A: The dataset includes data through 2024. Update by re-running the notebook with new data.

**Q: Can I use this for research?**
A: Yes, but cite appropriately and note it's for educational purposes.

---

## 📞 Support

For questions or issues:
1. Check this guide first
2. Review the README.md
3. Contact the team members
4. Check Jupyter notebook documentation

---

**Enjoy exploring life expectancy predictions! 🌍📊**
