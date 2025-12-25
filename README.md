# Liver Disease Prediction using Machine Learning

## Project Overview
Machine learning system to predict liver disease from blood test results. Compares multiple algorithms using the Indian Liver Patient Dataset.

## Results
- **Best Model**: Logistic Regression
- **Accuracy**: 73.5%
- **Recall**: 96.4%
- **Top Features**: ALT (Sgpt), AST (Sgot), Direct Bilirubin

## Project Structure
- `liver_disease_analysis.ipynb` - Complete analysis notebook
- `src/` - Data loading and model training code
- `models/` - Trained model and scaler
- `reports/` - Visualizations and results

## Quick Start
```bash
pip install -r requirements.txt
python -c "from src.model_trainer import train_and_evaluate_models; results = train_and_evaluate_models()"
