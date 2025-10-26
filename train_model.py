import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
import sys
import os
import json
import joblib # For saving the model

def train_and_evaluate(base_folder_path):
    """
    Loads the cleaned data, trains a Random Forest model,
    and returns a dictionary of evaluation metrics.
    """
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    results = {}
    
    # === 1. Define Paths ===
    processed_folder = os.path.join(base_folder_path, "data", "processed")
    logs_folder = os.path.join(base_folder_path, "logs")
    csv_path = os.path.join(processed_folder, "master_dataset_cleaned.csv")
    model_path = os.path.join(logs_folder, "random_forest_model.pkl")

    # === 2. Load dataset ===
    try:
        df = pd.read_csv(csv_path)
        results['data_load'] = f"✅ Successfully loaded 'master_dataset_cleaned.csv' ({len(df)} rows)."
    except FileNotFoundError:
        return {"error": f"❌ Error: '{csv_path}' not found."}
    except pd.errors.EmptyDataError:
        return {"error": f"❌ Error: '{csv_path}' is empty. No data to train on."}
    except Exception as e:
        return {"error": f"❌ Error loading data: {str(e)}"}

    # === 3. Create a proper Date column for aggregation ===
    df_agg = df.copy()
    
    # Create the Date column for grouping
    try:
        df_agg['Date'] = pd.to_datetime(
            df_agg['Year'].astype(str) + '-' + df_agg['Month'].astype(str) + '-01'
        )
    except Exception as e:
        return {"error": f"❌ Error creating 'Date' column: {e}"}

    # === 4. Features and target ===
    X = df[["Year", "Month", "Consultation_Type", "Case", "Sex", "Age_range"]]
    y = df["Total"]

    # === 5. Split data (80/20) ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    results['data_split'] = f"Train: {len(X_train)} rows | Test: {len(X_test)} rows"

    # === 6. Preprocess categorical and numeric features ===
    # All features listed are categorical except Year and Month
    categorical_features = ["Consultation_Type", "Case", "Sex", "Age_range"]
    numeric_features = ["Year", "Month"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features)
        ],
        remainder="drop"
    )

    # === 7. Model pipeline ===
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    # === 8. Train model ===
    model.fit(X_train, y_train)

    # === 9. Evaluate model (on test set) ===
    y_pred_test = model.predict(X_test)

    results['r2'] = r2_score(y_test, y_pred_test)
    results['mae'] = mean_absolute_error(y_test, y_pred_test)
    results['mse'] = mean_squared_error(y_test, y_pred_test)
    results['rmse'] = np.sqrt(results['mse'])

    # === 10. Generate predictions for the entire dataset ===
    full_predictions = model.predict(X)
    df_agg['Predicted'] = full_predictions

    # === 11. Aggregate data by month ===
    df_summary = df_agg.groupby('Date').agg(
        Actual_Total_Patients=('Total', 'sum'),
        Predicted_Total_Patients=('Predicted', 'sum')
    ).reset_index().sort_values('Date')

    # Format and save last 5 predictions
    df_summary['Predicted_Total_Patients'] = df_summary['Predicted_Total_Patients'].round(0).astype(int)
    df_summary['Date'] = df_summary['Date'].dt.strftime('%Y-%m')
    results['summary_table_tail'] = df_summary.tail(5).to_dict('records')

    # === 12. Save the model ===
    try:
        os.makedirs(logs_folder, exist_ok=True)
        joblib.dump(model, model_path)
        results['model_saved'] = f"✅ Model saved to {model_path}"
    except Exception as e:
        results['model_saved'] = f"❌ Error saving model: {str(e)}"
    
    return results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        # Default for local testing, assuming script is in D:\FlaskDev
        base_path = r"D:\FlaskDev" 
        
    final_results = train_and_evaluate(base_path)
    
    # --- IMPORTANT ---
    # Print the final JSON results to stdout
    # The Flask app will read this output.
    print(json.dumps(final_results, indent=2))
