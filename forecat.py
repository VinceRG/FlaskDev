# {file: train_forecast_model.py}
# This script includes an 80/20 split for evaluation and only displays results

import pandas as pd
import numpy as np
import warnings
import sys
import json
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def train_sarima_model(base_folder_path):
    """
    Loads and aggregates data, performs an 80/20 train-test split for evaluation,
    calculates metrics, and displays results without saving anything.
    """
    
    warnings.filterwarnings('ignore')
    results = {}
    
    # === 1. Define Paths ===
    processed_folder = base_folder_path + "/data/processed"
    csv_path = processed_folder + "/master_dataset_cleaned.csv"

    # === 2. Load and Aggregate Data ===
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return {"error": "master_dataset_cleaned.csv is empty."}
            
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
        ts_data = df.groupby('Date')['Total'].sum().asfreq('MS').fillna(0) # 'MS' = Month Start
        
        if len(ts_data) < 12: # SARIMA needs at least one full seasonal cycle
             return {"error": "Not enough data for SARIMA (less than 12 months)."}
        
    except Exception as e:
        return {"error": f"Error loading or preparing data: {str(e)}"}

    # === 3. Split the Data (80/20 sequential) ===
    split_point = int(len(ts_data) * 0.8)
    train_data = ts_data.iloc[:split_point]
    test_data = ts_data.iloc[split_point:]

    if len(test_data) == 0:
        return {"error": "Test set is empty. Not enough data (less than 2 years) to split 80/20."}
        
    results['data_split'] = f"Train: {len(train_data)} months, Test: {len(test_data)} months"

    # === 4. Train the SARIMA Model (on training data) ===
    try:
        # These are standard starting parameters for seasonal (12-month) data
        model = SARIMAX(train_data,
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 12),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        
        model_fit = model.fit(disp=False)
        
    except Exception as e:
        return {"error": f"Error training model: {str(e)}"}

    # === 5. Make Predictions (for the test set period) ===
    # We forecast for the number of steps equal to the test set size
    predictions = model_fit.forecast(steps=len(test_data))
    predictions.index = test_data.index # Align index for comparison

    # === 6. Calculate Metrics ===
    try:
        results['r2'] = round(r2_score(test_data, predictions), 4)
        results['mae'] = round(mean_absolute_error(test_data, predictions), 2)
        results['mse'] = round(mean_squared_error(test_data, predictions), 2)
        results['rmse'] = round(np.sqrt(results['mse']), 2)
    except Exception as e:
        results['metrics_error'] = str(e)

    # === 7. Generate and Display Plot ===
    try:
        plt.figure(figsize=(12, 7))
        plt.plot(train_data, label='Training Data', color='blue', linewidth=2)
        plt.plot(test_data, label='Actual Test Data', color='orange', linewidth=2)
        plt.plot(predictions, label='SARIMA Forecast', color='green', linestyle='--', linewidth=2)
        plt.title(f'SARIMA Model Evaluation (80/20 Split)\nR²: {results.get("r2", 0):.4f} | RMSE: {results.get("rmse", 0):.2f}')
        plt.xlabel('Date')
        plt.ylabel('Total Patients')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()  # Display the plot
        results['plot_displayed'] = "Chart displayed successfully"
    except Exception as e:
        results['plot_error'] = f"Could not display plot: {str(e)}"

    # === 8. Print Detailed Results ===
    print("\n" + "="*60)
    print("SARIMA MODEL EVALUATION RESULTS (80/20 Split)")
    print("="*60)
    print(f"Data Split: {results.get('data_split', 'N/A')}")
    print(f"R² Score: {results.get('r2', 'N/A')}")
    print(f"Mean Absolute Error (MAE): {results.get('mae', 'N/A')}")
    print(f"Mean Squared Error (MSE): {results.get('mse', 'N/A')}")
    print(f"Root Mean Squared Error (RMSE): {results.get('rmse', 'N/A')}")
    print("="*60)
    
    # Interpretation of results
    r2 = results.get('r2', 0)
    if r2 > 0.8:
        interpretation = "Excellent fit"
    elif r2 > 0.6:
        interpretation = "Good fit"
    elif r2 > 0.4:
        interpretation = "Moderate fit"
    else:
        interpretation = "Poor fit"
    
    print(f"Model Performance: {interpretation}")
    print("="*60)
    
    return results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = r"D:\FlaskDev" 
        
    final_results = train_sarima_model(base_path)
    
    # # Print JSON results for potential programmatic use
    # print("\nJSON Results:")
    # print(json.dumps(final_results, indent=2))