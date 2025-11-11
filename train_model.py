import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance # --- NEW ---
import warnings
import sys
import os
import json
import joblib

def train_and_evaluate(base_folder_path):
    """
    Loads the cleaned data, trains a Random Forest model,
    saves the model, evaluation metrics, and feature importance.
    """
    warnings.filterwarnings('ignore')
    results = {}

    # === 1. Define Paths ===
    processed_folder = os.path.join(base_folder_path, "data", "processed")
    logs_folder = os.path.join(base_folder_path, "logs")
    csv_path = os.path.join(processed_folder, "master_dataset_cleaned.csv")
    model_path = os.path.join(logs_folder, "random_forest_model.pkl")
    eval_path = os.path.join(logs_folder, "model_evaluation.json")
    importance_path = os.path.join(logs_folder, "feature_importance.json") # --- NEW ---

    # === 2. Load dataset ===
    try:
        df = pd.read_csv(csv_path)
        results['data_load'] = f"✅ Loaded dataset ({len(df)} rows)"
    except FileNotFoundError:
        return {"error": f"❌ '{csv_path}' not found."}
    except pd.errors.EmptyDataError:
        return {"error": f"❌ '{csv_path}' is empty."}
    except Exception as e:
        return {"error": f"❌ Error loading data: {e}"}

    # === 3. Create Date column for grouping ===
    try:
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
    except Exception as e:
        return {"error": f"❌ Error creating 'Date' column: {e}"}

    # === 4. Features and target ===
    # --- UPDATED: Ensure 'Case' is included as it's in your preprocessor ---
    features_list = ["Year", "Month", "Consultation_Type", "Case", "Sex", "Age_range"]
    X = df[features_list]
    y = df["Total"]

    # === 5. Train/Test Split ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results['data_split'] = f"Train: {len(X_train)} rows | Test: {len(X_test)} rows"

    # === 6. Preprocessing ===
    categorical_features = ["Consultation_Type", "Case", "Sex", "Age_range"]
    numeric_features = ["Year", "Month"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ])

    # === 7. Model Pipeline ===
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    # === 8. Train Model ===
    model.fit(X_train, y_train)

    # === 9. Evaluate Model ===
    y_pred_test = model.predict(X_test)

    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)

    results['evaluation'] = {
        "R² Score": round(r2, 4),
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4)
    }
    
    # --- NEW: 9b. Calculate and Save Feature Importance ---
    try:
        print("\n⏳ Calculating feature importance (this may take a moment)...")
        perm_importance = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        
        # Create a DataFrame for easy viewing and saving
        df_importance = pd.DataFrame(
            {'Feature': features_list, 'Importance': perm_importance.importances_mean}
        )
        df_importance.sort_values(by='Importance', ascending=False, inplace=True)
        df_importance['Importance'] = df_importance['Importance'].round(4)

        print("\n📈 Model Feature Importance:")
        print(df_importance.to_string(index=False))

        # Save importance to JSON
        with open(importance_path, "w") as f:
            json.dump(df_importance.to_dict('records'), f, indent=2)
        results['feature_importance_saved'] = f"✅ Feature importance saved to {importance_path}"

    except Exception as e:
        print(f"❌ Error calculating feature importance: {e}")
        results['feature_importance_saved'] = f"❌ Error: {e}"
    # --- END NEW ---

    # === 10. Predict entire dataset ===
    full_predictions = model.predict(X)
    df['Predicted'] = full_predictions

    df_summary = df.groupby('Date').agg(
        Actual_Total_Patients=('Total', 'sum'),
        Predicted_Total_Patients=('Predicted', 'sum')
    ).reset_index().sort_values('Date')

    df_summary['Predicted_Total_Patients'] = df_summary['Predicted_Total_Patients'].round(0).astype(int)
    df_summary['Date'] = df_summary['Date'].dt.strftime('%Y-%m')
    results['summary_table_tail'] = df_summary.tail(5).to_dict('records')

    # === 11. Save Model ===
    try:
        os.makedirs(logs_folder, exist_ok=True)
        joblib.dump(model, model_path)
        results['model_saved'] = f"✅ Model saved to {model_path}"
    except Exception as e:
        results['model_saved'] = f"❌ Error saving model: {e}"

    # === 12. Save Evaluation Results ===
    try:
        with open(eval_path, "w") as f:
            json.dump(results, f, indent=2)
        results['evaluation_saved'] = f"✅ Evaluation metrics saved to {eval_path}"
    except Exception as e:
        results['evaluation_saved'] = f"❌ Error saving evaluation file: {e}"

    # === 13. Print Evaluation Summary ===
    print("\n📊 Model Accuracy and Error Metrics:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = r"D:\FlaskDev"

    # Train the model
    train_and_evaluate(base_path)

    # Only display success message
    print(json.dumps({"message": "Model trained successfully"}))