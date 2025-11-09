#!/bin/bash

echo "🌍 Life Expectancy ML Web Application"
echo "======================================"
echo ""

# Check if data files exist
if [ ! -f "data/world_bank_data_cleaned.csv" ]; then
    echo "⚠️  Data files not found!"
    echo "Please run the Life_Expectancy_ML_Pipeline.ipynb notebook first."
    echo ""
    echo "Steps:"
    echo "1. Open Jupyter: jupyter notebook"
    echo "2. Run Life_Expectancy_ML_Pipeline.ipynb (all cells)"
    echo "3. Then run this script again"
    exit 1
fi

echo "✓ Data files found"
echo ""
echo "Starting Streamlit application..."
echo "The app will open in your browser automatically."
echo ""
echo "To stop the app, press Ctrl+C"
echo ""

streamlit run app.py
