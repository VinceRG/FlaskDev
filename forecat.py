import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

print("📊 Random Forest Outbreak Factor Analysis Started...\n")

# === 1. Load dataset ===
try:
    df = pd.read_csv("master_dataset_cleaned.csv")
    print(f"✅ Loaded dataset with {len(df)} rows.")
except FileNotFoundError:
    print("❌ Error: 'master_dataset_cleaned.csv' not found in current directory.")
    exit()

# === 2. Preprocess Data ===
df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01")

# === 3. Define features and target ===
features = ["Year", "Month", "Consultation_Type", "Case", "Sex", "Age_range"]
target = "Total"

X = df[features]
y = df[target]

# === 4. Split dataset ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"🔹 Training: {len(X_train)} rows | Testing: {len(X_test)} rows\n")

# === 5. Preprocessing pipeline ===
categorical_features = ["Consultation_Type", "Case", "Sex", "Age_range"]
numeric_features = ["Year", "Month"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# === 6. Model setup ===
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

# === 7. Train the model ===
print("🤖 Training Random Forest model...")
model.fit(X_train, y_train)
print("✅ Training complete.\n")

# === 8. Evaluate ===
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("📈 Model Evaluation Metrics:")
print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")
print(f"RMSE     : {rmse:.2f}\n")

# === 9. Feature Importance ===
rf_model = model.named_steps['regressor']
encoder = model.named_steps['preprocessor'].named_transformers_['cat']
cat_features = encoder.get_feature_names_out(categorical_features)
all_features = np.concatenate([cat_features, numeric_features])
importances = rf_model.feature_importances_

feat_imp = pd.DataFrame({'Feature': all_features, 'Importance': importances})
feat_imp = feat_imp.sort_values('Importance', ascending=False).head(15)
print("🔥 Top 15 Most Influential Factors on ER Admissions:")
print(feat_imp.to_string(index=False))

# === 10. Predict over time for visualization ===
df["Predicted"] = model.predict(X)

# Aggregate monthly totals
df_summary = df.groupby("Date").agg(
    Actual_Total=('Total', 'sum'),
    Predicted_Total=('Predicted', 'sum')
).reset_index()

# === 11. Detect Outbreak Months Automatically ===
mean_total = df_summary["Actual_Total"].mean()
std_total = df_summary["Actual_Total"].std()
threshold = mean_total + 1.5 * std_total

outbreak_months = df_summary[df_summary["Actual_Total"] > threshold]
print("\n🚨 Detected Outbreak Months (High Admission Spikes):")
if outbreak_months.empty:
    print("No outbreak months detected.")
else:
    print(outbreak_months[["Date", "Actual_Total"]].to_string(index=False))

# === 12. Identify Top Cases Causing Each Outbreak ===
if not outbreak_months.empty:
    print("\n📊 Top Contributing Cases During Outbreak Months:")
    for date in outbreak_months["Date"]:
        month_cases = df[df["Date"] == date].groupby("Case")["Total"].sum().sort_values(ascending=False).head(5)
        print(f"\n🗓️ {date.strftime('%B %Y')}:")
        print(month_cases.to_string())

        # Plot each outbreak's top cases
        month_cases.plot(kind="bar", figsize=(7, 4), color="tomato")
        plt.title(f"Top Cases During {date.strftime('%B %Y')} Outbreak")
        plt.ylabel("Total ER Admissions")
        plt.xlabel("Case Type")
        plt.tight_layout()
        plt.show()

# === 13. Plot Actual vs Predicted ===
plt.figure(figsize=(10, 5))
plt.plot(df_summary["Date"], df_summary["Actual_Total"], label="Actual ER Admissions", marker='o')
plt.plot(df_summary["Date"], df_summary["Predicted_Total"], label="Predicted ER Admissions", linestyle="--", marker='x')
plt.axhline(y=threshold, color="red", linestyle=":", label="Outbreak Threshold")
plt.title("Actual vs Predicted ER Admissions Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Patients")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 14. Feature Importance Chart ===
plt.figure(figsize=(8, 5))
plt.barh(feat_imp["Feature"], feat_imp["Importance"], color="purple")
plt.title("Top 15 Most Influential Factors")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\n✅ Outbreak detection and analysis complete! Charts displayed successfully.")
