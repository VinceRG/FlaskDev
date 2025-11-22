import os
import sys
import subprocess
from unittest import result
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import json
import joblib 
import warnings 
import numpy as np 
from flask import send_from_directory
# --- ADDED IMPORTS for your new route ---
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- Configuration ---
BASE_FOLDER = r"D:\FlaskDev"
INPUT_FOLDER = os.path.join(BASE_FOLDER, "data", "excel_folder")
PROCESSED_FOLDER = os.path.join(BASE_FOLDER, "data", "processed")
LOGS_FOLDER = os.path.join(BASE_FOLDER, "logs")
CLEANED_CSV = os.path.join(PROCESSED_FOLDER, "master_dataset_cleaned.csv")
EXCEL_LOG = os.path.join(LOGS_FOLDER, "converted_files.txt")
MODEL_FILE = os.path.join(LOGS_FOLDER, "random_forest_model.pkl") 
CASE_DICT_FILE = os.path.join(LOGS_FOLDER, "case_dictionary.json") 
MONTH_CASE_SUMMARY_FILE = os.path.join(LOGS_FOLDER, "monthly_case_summary.csv")

ALLOWED_EXTENSIONS = {'xlsx'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = INPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'your-secret-key-goes-here' 

# --- RESOURCE CONFIG ---
SHIFTS_PER_DAY = 2          # Morning + night
NURSES_PER_SHIFT = 7        # 7 nurses per shift
DAYS_PER_MONTH = 30         # Avg days in a month

RESOURCE_INVENTORY = {
    # -----------------------------
    # CONSUMABLES (per shift capacity)
    # -----------------------------
    "Temperature Probe Cover": {
        "unit": "pcs",
        "type": "consumable",
        # assume 120 probe covers available per shift
        "capacity_per_shift": 120
    },
    "BP Cuff Disposable Cover": {
        "unit": "pcs",
        "type": "consumable",
        "capacity_per_shift": 120
    },
    "Stethoscope Earpiece Cover": {
        "unit": "pcs",
        "type": "consumable",
        "capacity_per_shift": 120
    },
    "HGT Strips": {
        "unit": "strips",
        "type": "consumable",
        # from reference: 100 strips/shift
        "capacity_per_shift": 100
    },
    "Nebulizer Kit": {
        "unit": "kits",
        "type": "consumable",
        # from reference: 50/shift
        "capacity_per_shift": 50
    },
    "Oxygen Mask": {
        "unit": "pcs",
        "type": "consumable",
        # Oxygen cannula/mask: 50/shift
        "capacity_per_shift": 50
    },
    "Nebulizer Mask": {
        "unit": "pcs",
        "type": "consumable",
        # align with Nebulizer kit
        "capacity_per_shift": 50
    },
    "Micropore Tape": {
        "unit": "rolls",
        "type": "consumable",
        # not specified; set modest stock per shift
        "capacity_per_shift": 10
    },
    "IV Cannula": {
        "unit": "pcs",
        "type": "consumable",
        # 100 each for 20/22/24/26 → ~400/shift
        "capacity_per_shift": 400
    },
    "Syringe": {
        "unit": "pcs",
        "type": "consumable",
        # 1/3/5/10cc 100/shift + 20/50cc 20/shift ≈ 120
        "capacity_per_shift": 120
    },
    "Soluset": {
        "unit": "sets",
        "type": "consumable",
        # 50/shift
        "capacity_per_shift": 50
    },
    "Microset": {
        "unit": "sets",
        "type": "consumable",
        # 50/shift
        "capacity_per_shift": 50
    },
    "Macroset": {
        "unit": "sets",
        "type": "consumable",
        # 50/shift
        "capacity_per_shift": 50
    },

    # (optional) extra consumables from your list
    "Elastic Bandage": {
        "unit": "pcs",
        "type": "consumable",
        "capacity_per_shift": 5
    },
    "Arm Sling": {
        "unit": "pcs",
        "type": "consumable",
        "capacity_per_shift": 5
    },
    "Sterile Gloves": {
        "unit": "pairs",
        "type": "consumable",
        # 10/size/shift → simplify to total ~30 pairs
        "capacity_per_shift": 30
    },

    # -----------------------------
    # EQUIPMENT (concurrent capacity)
    # -----------------------------
    "Cardiac Monitor": {
        "unit": "units",
        "type": "equipment",
        # 8 wall-mount + 7 portable = 15
        "capacity_per_shift": 15
    },
    "Oxygen Tank": {
        "unit": "units",
        "type": "equipment",
        # 15 tanks with gauge
        "capacity_per_shift": 15
    },
    "Infusion Pump": {
        "unit": "units",
        "type": "equipment",
        # 5 infusion pumps
        "capacity_per_shift": 5
    },
    "Portable Suction Machine": {
        "unit": "units",
        "type": "equipment",
        # 8 wall-mount + 2 portable = 10 suction points
        "capacity_per_shift": 10
    },
    "ECG Machine": {
        "unit": "units",
        "type": "equipment",
        # 1 ECG machine
        "capacity_per_shift": 1
    },
    "Fetal Doppler": {
        "unit": "units",
        "type": "equipment",
        # 1 Doppler
        "capacity_per_shift": 1
    },
    "Gloves (Bedside)": {
        "unit": "boxes",
        "type": "equipment",
        # bedside boxes at stations; assume 10 boxes/shift
        "capacity_per_shift": 10
    },
    "Pulse Oximeter": {
        "unit": "units",
        "type": "equipment",
        # 2 portable pulse oximeters
        "capacity_per_shift": 2
    },
    "Nebulizer Machine": {
        "unit": "units",
        "type": "equipment",
        # 4 nebulizers
        "capacity_per_shift": 4
    },
    "Thermometer": {
        "unit": "units",
        "type": "equipment",
        # 5 thermometers
        "capacity_per_shift": 5
    },
    "Defibrillator": {
        "unit": "units",
        "type": "equipment",
        # 2 defibs
        "capacity_per_shift": 2
    },
    "Glucose Monitor": {
        "unit": "units",
        "type": "equipment",
        # 2 HGT machines
        "capacity_per_shift": 2
    },

    # -----------------------------
    # PER-NURSE CONSUMABLES
    # -----------------------------
    "Gown": {
        "unit": "pcs",
        "type": "consumable_per_nurse",
        # 1 gown per nurse per shift → 7
        "capacity_per_shift": 7
    },
    "N95 Mask": {
        "unit": "pcs",
        "type": "consumable_per_nurse",
        # 1 N95 per nurse per shift
        "capacity_per_shift": 7
    },

    # -----------------------------
    # FIXED SUPPLIES
    # -----------------------------
    "Stretcher": {
        "unit": "units",
        "type": "fixed",
        # 15 stretchers
        "total_available": 15
    },
    "Wheelchair": {
        "unit": "units",
        "type": "fixed",
        # 4 wheelchairs
        "total_available": 4
    },
    "Bassinet": {
        "unit": "units",
        "type": "fixed",
        # 2 bassinets
        "total_available": 2
    },
    "Crib": {
        "unit": "units",
        "type": "fixed",
        # 1 crib
        "total_available": 1
    },
    "Spine Board": {
        "unit": "units",
        "type": "fixed",
        "total_available": 1
    },
    "Cervical Collar": {
        "unit": "units",
        "type": "fixed",
        "total_available": 2
    },
    "Crash Cart": {
        "unit": "units",
        "type": "fixed",
        "total_available": 2
    },
    "Mayo Table": {
        "unit": "units",
        "type": "fixed",
        "total_available": 2
    },
    "Ambu Bag Mask": {
        "unit": "units",
        "type": "fixed",
        "total_available": 2
    },
    "Intubation Set": {
        "unit": "sets",
        "type": "fixed",
        "total_available": 4
    }
}


GENERAL_RESOURCE_USAGE = {
    # -----------------------------
    # Consumables (units per patient visit)
    # -----------------------------
    "Temperature Probe Cover": {
        "type": "consumable",
        "avg_use": 1.0   # almost every patient
    },
    "BP Cuff Disposable Cover": {
        "type": "consumable",
        "avg_use": 1.0
    },
    "Stethoscope Earpiece Cover": {
        "type": "consumable",
        "avg_use": 1.0
    },
    "HGT Strips": {
        "type": "consumable",
        # not every patient; approx 30%
        "avg_use": 0.3
    },
    "Nebulizer Kit": {
        "type": "consumable",
        # maybe 15–20% of ER patients need neb
        "avg_use": 0.18
    },
    "Oxygen Mask": {
        "type": "consumable",
        # 10–15% need oxygen
        "avg_use": 0.12
    },
    "Nebulizer Mask": {
        "type": "consumable",
        "avg_use": 0.18
    },
    "Micropore Tape": {
        "type": "consumable",
        # fraction of a roll per patient on average
        "avg_use": 0.05
    },
    "IV Cannula": {
        "type": "consumable",
        # maybe 30% of ER visits need IV access
        "avg_use": 0.3
    },
    "Syringe": {
        "type": "consumable",
        # around 1 syringe per patient on average
        "avg_use": 1.0
    },
    "Soluset": {
        "type": "consumable",
        "avg_use": 0.10
    },
    "Microset": {
        "type": "consumable",
        "avg_use": 0.10
    },
    "Macroset": {
        "type": "consumable",
        "avg_use": 0.12
    },
    "Elastic Bandage": {
        "type": "consumable",
        "avg_use": 0.02
    },
    "Arm Sling": {
        "type": "consumable",
        "avg_use": 0.02
    },
    "Sterile Gloves": {
        "type": "consumable",
        # used for procedures; small subset of patients
        "avg_use": 0.05
    },

    # -----------------------------
    # Equipment (concurrent usage rate per patient)
    # -----------------------------
    "Cardiac Monitor": {
        "type": "equipment",
        # about 5% of patients on monitor at any given time
        "avg_use": 0.05
    },
    "Oxygen Tank": {
        "type": "equipment",
        "avg_use": 0.06
    },
    "Infusion Pump": {
        "type": "equipment",
        "avg_use": 0.04
    },
    "Portable Suction Machine": {
        "type": "equipment",
        "avg_use": 0.04
    },
    "ECG Machine": {
        "type": "equipment",
        # used sequentially, not all day on one patient
        "avg_use": 0.01
    },
    "Fetal Doppler": {
        "type": "equipment",
        "avg_use": 0.01
    },
    "Gloves (Bedside)": {
        "type": "equipment",
        # rough proxy: fraction of boxes "open" in use
        "avg_use": 0.10
    },
    "Pulse Oximeter": {
        "type": "equipment",
        # some already monitored via cardiac monitor; fewer portables
        "avg_use": 0.04
    },
    "Nebulizer Machine": {
        "type": "equipment",
        "avg_use": 0.04
    },
    "Thermometer": {
        "type": "equipment",
        # shared across bays; low concurrent demand
        "avg_use": 0.05
    },
    "Defibrillator": {
        "type": "equipment",
        "avg_use": 0.005  # very occasionally in active use
    },
    "Glucose Monitor": {
        "type": "equipment",
        "avg_use": 0.01
    },

    # -----------------------------
    # Per-nurse consumables
    # -----------------------------
    "Gown": {
        "type": "consumable_per_nurse",
        "avg_use": 1.0  # 1 per nurse per shift
    },
    "N95 Mask": {
        "type": "consumable_per_nurse",
        "avg_use": 1.0
    },

    # -----------------------------
    # Fixed supplies (concurrent usage vs capacity)
    # -----------------------------
    "Stretcher": {
        "type": "fixed",
        # ~10–15% of patients on stretchers at once
        "avg_use": 0.15
    },
    "Wheelchair": {
        "type": "fixed",
        "avg_use": 0.05
    },
    "Bassinet": {
        "type": "fixed",
        "avg_use": 0.03
    },
    "Crib": {
        "type": "fixed",
        "avg_use": 0.03
    },
    "Spine Board": {
        "type": "fixed",
        "avg_use": 0.01
    },
    "Cervical Collar": {
        "type": "fixed",
        "avg_use": 0.01
    },
    "Crash Cart": {
        "type": "fixed",
        "avg_use": 0.005
    },
    "Mayo Table": {
        "type": "fixed",
        "avg_use": 0.01
    },
    "Ambu Bag Mask": {
        "type": "fixed",
        "avg_use": 0.01
    },
    "Intubation Set": {
        "type": "fixed",
        "avg_use": 0.01
    }
}


# --- Helper Functions ---
def get_top_10_by_type(df_month, consult_type_id, inv_case_dict_numeric, column_to_sum='Total'):
    """
    Helper to get top 10 cases for a specific consultation type,
    summing a specified column (e.g., 'Total' or 'Predicted_Total').
    """
    
    # Filter for the specific consultation type
    df_type = df_month[df_month['Consultation_Type'] == consult_type_id]
    
    if df_type.empty:
        return {'table': [], 'chart_data': {'labels': [], 'data': []}}
    
    # Group by 'Case' and sum the specified column
    top_10 = df_type.groupby('Case')[column_to_sum].sum().nlargest(10).reset_index()
    
    # Rename the summed column back to 'Total' so the frontend can read it
    top_10.rename(columns={column_to_sum: 'Total'}, inplace=True)

    # Map Case IDs to names
    top_10['CaseName'] = top_10['Case'].map(inv_case_dict_numeric).fillna('Unknown Case')
    
    # Format for table and chart
    table_data = top_10.to_dict('records')
    chart_labels = top_10['CaseName'].tolist()
    chart_data = top_10['Total'].tolist()
    
    return {
        'table': table_data,
        'chart_data': {
            'labels': chart_labels,
            'data': chart_data
        }
    }

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_dashboard_stats():
    stats = {
        'total_files': 0,
        'total_records': 0,
        'success_rate': 100.0,
        'processing': 0 
    }
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    os.makedirs(LOGS_FOLDER, exist_ok=True)
    try:
        if os.path.exists(EXCEL_LOG):
            with open(EXCEL_LOG, "r") as f:
                processed_count = len(f.read().splitlines())
        stats['total_files'] = processed_count
    except Exception:
        stats['total_files'] = 0
    try:
        df = pd.read_csv(CLEANED_CSV)
        stats['total_records'] = len(df)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        stats['total_records'] = 0
    except Exception as e:
        print(f"Error reading cleaned CSV: {e}")
        stats['total_records'] = 0
    if stats['total_files'] == 0:
        stats['success_rate'] = 0.0
    else:
        stats['success_rate'] = 100.0
    return stats
    
# --- Flask Routes ---
@app.route('/')
def index():
    stats = get_dashboard_stats()
    return render_template('index.html', stats=stats)

@app.route('/upload', methods=['POST'])
def upload_file():
    files = request.files.getlist('file')
    if not files or files[0].filename == '':
        flash('No selected files', 'error')
        return redirect(url_for('index'))
    processed_files_log = set()
    try:
        if os.path.exists(EXCEL_LOG):
            with open(EXCEL_LOG, "r") as f:
                processed_files_log = set(line.strip() for line in f.readlines())
    except Exception as e:
        print(f"Warning: Could not read log file {EXCEL_LOG}. {e}")
    new_files_saved = []
    skipped_files_duplicate = []
    invalid_type_files = []
    for file in files:
        filename = secure_filename(file.filename)
        if filename == '': continue
        if not allowed_file(filename):
            invalid_type_files.append(filename)
            continue
        if filename in processed_files_log:
            skipped_files_duplicate.append(filename)
            continue
        try:
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            new_files_saved.append(filename)
        except Exception as e:
            print(f"Error saving file {filename}: {e}")
            invalid_type_files.append(f"{filename} (Save Error)")
    pipeline_error = None
    if new_files_saved:
        try:
            print("--- [APP] Running data pipeline ---")
            pipeline_path = os.path.join(BASE_FOLDER, 'pipeline.py') 
            proc_env = os.environ.copy()
            proc_env['PYTHONIOENCODING'] = 'utf-8'
            result = subprocess.run(
                [sys.executable, pipeline_path, BASE_FOLDER],
                capture_output=True, text=True, check=True,
                encoding='utf-8', env=proc_env
            )
            print("--- [APP] Pipeline STDOUT:", result.stdout)
            print("--- [APP] Pipeline STDERR:", result.stderr)
            print("--- [APP] Pipeline complete. ---")
        except subprocess.CalledProcessError as e:
            print(f"--- [APP] Pipeline failed with code {e.returncode} ---")
            print("Pipeline STDOUT:", e.stdout)
            print("Pipeline STDERR:", e.stderr)
            pipeline_error = f'Data pipeline failed. See console.'
    if pipeline_error:
        flash(f'Uploaded {len(new_files_saved)} file(s), but the pipeline failed: {pipeline_error}', 'error')
    elif new_files_saved:
        flash(f'Successfully processed {len(new_files_saved)} new file(s). Data is ready for training.', 'success')
    if skipped_files_duplicate:
        flash(f'Skipped {len(skipped_files_duplicate)} file(s) (already processed): {", ".join(skipped_files_duplicate)}', 'warning')
    if invalid_type_files:
        flash(f'Skipped {len(invalid_type_files)} file(s) (invalid type or save error).', 'warning')
    if not new_files_saved and not skipped_files_duplicate and not invalid_type_files:
        flash('No files were selected.', 'error')
    elif not new_files_saved:
        flash('No new files to process.', 'info')
    return redirect(url_for('index'))

@app.route('/train', methods=['POST'])
def train_model():
    print("--- [APP] Received request to train model ---")
    try:
        train_script_path = os.path.join(BASE_FOLDER, 'train_model.py')
        proc_env = os.environ.copy()
        proc_env['PYTHONIOENCODING'] = 'utf-8'

        # Run training script
        result = subprocess.run(
            [sys.executable, train_script_path, BASE_FOLDER],
            capture_output=True, text=True, check=True,
            encoding='utf-8', env=proc_env
        )

        # Try parsing JSON output from the training script
        if result.returncode == 0:
            flash("✅ Model training completed successfully!", "success")
        else:
            flash("❌ Model training failed.", "error")

    except subprocess.CalledProcessError as e:
        print(f"--- [APP] Training script failed with code {e.returncode} ---")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        flash('❌ Model training failed. See server logs for details.', 'error')
    except Exception as e:
        print(f"--- [APP] Unexpected error: {e} ---")
        flash('❌ An unexpected error occurred. Check logs.', 'error')

    return redirect(url_for('index'))


# ---------------------------------------------------------------
# --- DASHBOARD ROUTES ---
# ---------------------------------------------------------------

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Predict total patients for a given Year–Month–Consultation_Type–Case
    using the aggregated monthly-per-case model.
    """
    try:
        if not os.path.exists(MODEL_FILE):
            return jsonify({'error': f'Model file not found at {MODEL_FILE}. Please train the model first.'}), 500

        model = joblib.load(MODEL_FILE)
        data = request.json or {}

        year = int(data['year'])
        month = int(data['month'])
        consult_type = int(data['consult_type'])
        case_id = int(data['case_id'])

        # We may receive sex/age_range from the frontend, but the new model
        # does NOT use them. They are safely ignored:
        # sex = int(data.get('sex', 0))
        # age_range = int(data.get('age_range', 0))

        X_new = pd.DataFrame({
            "Year": [year],
            "Month": [month],
            "Consultation_Type": [consult_type],
            "Case": [case_id]
        })

        prediction = model.predict(X_new)
        # prediction[0] is the predicted TOTAL PATIENTS for that combination in that month
        return jsonify({'prediction': round(float(prediction[0]), 2)})

    except KeyError as e:
        return jsonify({'error': f'Missing required field: {str(e)}'}), 400
    except FileNotFoundError:
        return jsonify({'error': f'Model file not found at {MODEL_FILE}. Please train the model first.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/past_data')
def api_past_data():
    try:
        df = pd.read_csv(CLEANED_CSV)
        table_data = df.tail(20).to_dict('records')
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
        chart_data = df.groupby('Date')['Total'].sum().reset_index()
        return jsonify({
            'table': table_data,
            'chart_data': {
                'labels': chart_data['Date'].dt.strftime('%Y-%m').tolist(),
                'data': chart_data['Total'].tolist()
            }
        })
    except FileNotFoundError:
        return jsonify({'error': f'Data file not found at {CLEANED_CSV}. Please process files first.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/available_years')
def api_available_years():
    try:
        df = pd.read_csv(CLEANED_CSV)
        if df.empty:
            current_year = pd.Timestamp.now().year
            years = list(range(current_year - 5, current_year + 6))
        else:
            min_year = int(df['Year'].min())
            max_year = int(df['Year'].max())
            years = list(range(min_year, max_year + 6))
        return jsonify({'years': sorted(years, reverse=True)})
    except FileNotFoundError:
        current_year = pd.Timestamp.now().year
        years = list(range(current_year - 5, current_year + 6))
        return jsonify({'years': sorted(years, reverse=True)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- MODIFIED API ENDPOINT FOR AGE/CASE ANALYSIS ---
@app.route('/api/age_case_analysis')
def api_age_case_analysis():
    """
    Analyzes the cleaned data to find the top case for EACH age range,
    filtered by the requested consultation type.
    """
    try:
        # --- 1. Get consultation type from URL parameter ---
        consult_type = request.args.get('consult_type', default=1, type=int)

        # --- 2. Load Data ---
        df = pd.read_csv(CLEANED_CSV)
        with open(CASE_DICT_FILE, 'r') as f:
            case_dict = json.load(f)
        
        # Invert dictionaries for mapping IDs to Names
        inv_case_dict = {v: k for k, v in case_dict.items()}
        
        # User-provided Age Range Map (inverted)
        age_map_raw = {
            "Under 1": 0, "1-4": 1, "5-9": 2, "10-14": 3, "15-18": 4, "19-24": 5,
            "25-29": 6, "30-34": 7, "35-39": 8, "40-44": 9, "45-49": 10, "50-54": 11,
            "55-59": 12, "60-64": 13, "65-69": 14, "70": 15, "70 Over": 15, "70 & OVER": 15
        }
        # Create a clean inverted map {0: "Under 1", 1: "1-4", ...}
        inv_age_map = {}
        # Use a list to preserve the intended order
        age_order_list = [
            ("Under 1", 0), ("1-4", 1), ("5-9", 2), ("10-14", 3), ("15-18", 4), ("19-24", 5),
            ("25-29", 6), ("30-34", 7), ("35-39", 8), ("40-44", 9), ("45-49", 10), ("50-54", 11),
            ("55-59", 12), ("60-64", 13), ("65-69", 14)
        ]
        
        # Handle the consolidated '70 & Over' group
        for name, idx in age_map_raw.items():
            if idx == 15:
                age_map_raw[name] = 15 # Ensure all '70...' map to 15
        
        inv_age_map = {idx: name for name, idx in age_order_list}
        inv_age_map[15] = "70 & Over" # Standardize "70 & Over"
        age_order_map = {name: i for i, (name, idx) in enumerate(age_order_list + [("70 & Over", 15)])}

        
        # --- 3. Perform Analysis ---
        # Filter for the requested consultation type
        df_filtered = df[df['Consultation_Type'] == consult_type].copy()
        
        if df_filtered.empty:
            # Return an empty list, but we'll format it with all age ranges
            # so the table doesn't look broken
            output_data = []
            for age_name, age_idx in age_order_map.items():
                 output_data.append({
                    "age_range": age_name,
                    "top_case": "N/A",
                    "total_patients": 0
                })
            return jsonify(output_data)

        # Group by Age_range and Case, sum up totals
        df_grouped = df_filtered.groupby(['Age_range', 'Case'])['Total'].sum().reset_index()

        # Find the index of the max 'Total' for each 'Age_range'
        df_top_cases = df_grouped.loc[df_grouped.groupby('Age_range')['Total'].idxmax()]

        # --- 4. Format Output ---
        # Create a dictionary of results for easy lookup
        results_dict = {}
        for _, row in df_top_cases.iterrows():
            age_range_name = inv_age_map.get(row['Age_range'])
            if age_range_name:
                results_dict[age_range_name] = {
                    "top_case": inv_case_dict.get(row['Case'], f"Unknown ({row['Case']})"),
                    "total_patients": int(row['Total'])
                }

        # Build the final list in the correct order
        output_data = []
        for age_name, _ in age_order_map.items():
            if age_name in results_dict:
                output_data.append({
                    "age_range": age_name,
                    "top_case": results_dict[age_name]["top_case"],
                    "total_patients": results_dict[age_name]["total_patients"]
                })
            else:
                # Add a row even if no data exists for that age range
                output_data.append({
                    "age_range": age_name,
                    "top_case": "N/A",
                    "total_patients": 0
                })

        return jsonify(output_data)

    except FileNotFoundError:
        return jsonify({'error': 'Data or case dictionary file not found. Please process files.'}), 404
    except Exception as e:
        print(f"Error in /api/age_case_analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/feature_importance')
def api_feature_importance():
    try:
        # Load the saved feature importance file
        with open(os.path.join(LOGS_FOLDER, 'feature_importance.json'), 'r') as f:
            importance_data = json.load(f)
        return jsonify(importance_data)
    except FileNotFoundError:
        return jsonify({'error': f'feature_importance.json not found. Please train the model first.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/top_cases', methods=['POST'])
def api_top_cases():
    """
    Gets top 10 cases for a selected month, categorized by consultation type,
    using predictions from the aggregated monthly-per-case Random Forest model.

    Uses logs/monthly_case_summary.csv generated by train_model.py:
    columns include:
        Year, Month, Consultation_Type, Case, Total_Patients, Predicted, Date
    """
    try:
        request_data = request.get_json() or {}
        month = int(request_data['month'])

        # --- 1. Load case dictionary ---
        if not os.path.exists(CASE_DICT_FILE):
            return jsonify({'error': 'Case dictionary file not found. Please process files.'}), 404

        with open(CASE_DICT_FILE, 'r') as f:
            case_dict = json.load(f)
        inv_case_dict_numeric = {v: k for k, v in case_dict.items()}

        # --- 2. Ensure summary file from the new model exists ---
        if not os.path.exists(MONTH_CASE_SUMMARY_FILE):
            return jsonify({'error': 'Monthly case summary not found. Please train the model first.'}), 500

        # --- 3. Load aggregated monthly-per-case data ---
        df = pd.read_csv(MONTH_CASE_SUMMARY_FILE)
        if 'Year' not in df.columns or 'Month' not in df.columns:
            return jsonify({'error': 'monthly_case_summary.csv has unexpected format (missing Year/Month).'}), 500


        # Expect columns: Year, Month, Consultation_Type, Case, Total_Patients, Predicted, Date
        if 'Month' not in df.columns or 'Consultation_Type' not in df.columns or \
           'Case' not in df.columns or 'Total_Patients' not in df.columns or \
           'Predicted' not in df.columns:
            return jsonify({'error': 'monthly_case_summary.csv has unexpected format.'}), 500

        # --- 4. Filter by selected month (across all years, to match previous behavior) ---
        df_month_all = df[df['Month'] == month].copy()
        if df_month_all.empty:
            return jsonify({
                'error': f'No aggregated data found for month {pd.to_datetime(f"2024-{month}-01").strftime("%B")}.'
            }), 404

        latest_year = df_month_all['Year'].max()
        df_month = df_month_all[df_month_all['Year'] == latest_year].copy()

        # --- 5. Compute overall totals & accuracy metrics for that month ---
        actual_total = df_month["Total_Patients"].sum()
        predicted_total = df_month["Predicted"].sum()

        # Accuracy metrics at aggregated level; guard against single-row edge case for R²
        if len(df_month) > 1:
            r2 = r2_score(df_month["Total_Patients"], df_month["Predicted"])
        else:
            r2 = float('nan')  # not defined for single sample

        mae = mean_absolute_error(df_month["Total_Patients"], df_month["Predicted"])
        rmse = np.sqrt(mean_squared_error(df_month["Total_Patients"], df_month["Predicted"]))

        # --- 6. Get categorized top 10 cases BASED ON PREDICTION ---
        # Note: column_to_sum='Predicted' now (not 'Predicted_Total')
        consultation_data = get_top_10_by_type(
            df_month, 1, inv_case_dict_numeric, column_to_sum='Predicted'
        )
        diagnosis_data = get_top_10_by_type(
            df_month, 2, inv_case_dict_numeric, column_to_sum='Predicted'
        )
        mortality_data = get_top_10_by_type(
            df_month, 3, inv_case_dict_numeric, column_to_sum='Predicted'
        )

        # --- 7. Return JSON response ---
        return jsonify({
            'month': month,
            'total_summary': {
                'actual_total': round(float(actual_total), 2),
                'predicted_total': round(float(predicted_total), 2),
                'accuracy_metrics': {
                    'R²': None if np.isnan(r2) else round(float(r2), 4),
                    'MAE': round(float(mae), 4),
                    'RMSE': round(float(rmse), 4)
                }
            },
            'consultation': consultation_data,
            'diagnosis': diagnosis_data,
            'mortality': mortality_data
        })

    except KeyError as e:
        return jsonify({'error': f'Missing required field in request: {str(e)}'}), 400
    except FileNotFoundError:
        return jsonify({'error': 'Data or case dictionary file not found. Please process files.'}), 404
    except Exception as e:
        print(f"Error in /api/top_cases: {e}")
        return jsonify({'error': str(e)}), 500

    
@app.route('/api/resource_needs', methods=['POST'])
def resource_needs():
    """
    Calculate monthly supply usage, equipment demand, and resource utilization.
    Expects JSON body with: predicted_visits, general_factor, nurses_per_shift
    """
    try:
        # Get data from JSON body (not query params)
        data = request.get_json()
        predicted_visits = float(data.get('predicted_visits', 0))
        general_factor = float(data.get('general_factor', 1.0))
        nurses_per_shift = int(data.get('nurses_per_shift', NURSES_PER_SHIFT))
        
        # Validate input
        if predicted_visits <= 0:
            return jsonify({'error': 'predicted_visits must be greater than 0'}), 400
        
        # Calculate adjusted monthly visits
        adjusted_visits = predicted_visits * general_factor
        
        # Calculate shifts per month
        shifts_per_month = SHIFTS_PER_DAY * DAYS_PER_MONTH
        avg_patients_per_shift = adjusted_visits / shifts_per_month
        
        results = []
        
        for resource_name, usage_info in GENERAL_RESOURCE_USAGE.items():
            inventory = RESOURCE_INVENTORY.get(resource_name)
            
            if not inventory:
                continue
            
            r_type = usage_info['type']
            avg_use = usage_info['avg_use']
            
            # -----------------------------
            # 1. CONSUMABLES (per patient)
            # -----------------------------
            if r_type == "consumable":
                # Monthly demand
                monthly_demand = adjusted_visits * avg_use
                
                # Monthly capacity
                capacity_per_shift = inventory['capacity_per_shift']
                monthly_capacity = capacity_per_shift * shifts_per_month
                
                # Utilization
                utilization = monthly_demand / monthly_capacity if monthly_capacity > 0 else 1
                
                # Status determination
                status = ("OVER_CAPACITY" if utilization > 1 else
                         "HIGH_USAGE" if utilization > 0.8 else
                         "MODERATE" if utilization > 0.5 else
                         "OK")
                
                results.append({
                    "resource_name": resource_name,
                    "type": "consumable",
                    "unit": inventory['unit'],
                    "monthly_demand": round(monthly_demand, 1),
                    "monthly_capacity": round(monthly_capacity, 1),
                    "utilization": round(utilization * 100, 1),  # as percentage
                    "status": status,
                    "shortage": round(max(0, monthly_demand - monthly_capacity), 1)
                })
            
            # -----------------------------
            # 2. EQUIPMENT (concurrent usage)
            # -----------------------------
            elif r_type == "equipment":
                # Average concurrent units needed per shift
                required_units_per_shift = avg_patients_per_shift * avg_use
                
                # Monthly "demand" = average concurrent need
                monthly_demand = required_units_per_shift
                
                # Capacity
                capacity_per_shift = inventory['capacity_per_shift']
                
                # Utilization
                utilization = required_units_per_shift / capacity_per_shift if capacity_per_shift > 0 else 1
                
                status = ("OVER_CAPACITY" if utilization > 1 else
                         "HIGH_USAGE" if utilization > 0.8 else
                         "MODERATE" if utilization > 0.5 else
                         "OK")
                
                results.append({
                    "resource_name": resource_name,
                    "type": "equipment",
                    "unit": inventory['unit'],
                    "monthly_demand": round(monthly_demand, 1),  # avg concurrent units needed
                    "monthly_capacity": capacity_per_shift,  # total available
                    "utilization": round(utilization * 100, 1),
                    "status": status,
                    "shortage": round(max(0, required_units_per_shift - capacity_per_shift), 1)
                })
            
            # -----------------------------
            # 3. PER-NURSE CONSUMABLES
            # -----------------------------
            elif r_type == "consumable_per_nurse":
                # Monthly demand based on nurses
                monthly_demand = nurses_per_shift * avg_use * shifts_per_month
                
                # Monthly capacity
                capacity_per_shift = inventory['capacity_per_shift']
                monthly_capacity = capacity_per_shift * shifts_per_month
                
                # Utilization
                utilization = monthly_demand / monthly_capacity if monthly_capacity > 0 else 1
                
                status = ("OVER_CAPACITY" if utilization > 1 else
                         "HIGH_USAGE" if utilization > 0.8 else
                         "MODERATE" if utilization > 0.5 else
                         "OK")
                
                results.append({
                    "resource_name": resource_name,
                    "type": "consumable_per_nurse",
                    "unit": inventory['unit'],
                    "monthly_demand": round(monthly_demand, 1),
                    "monthly_capacity": round(monthly_capacity, 1),
                    "utilization": round(utilization * 100, 1),
                    "status": status,
                    "nurses_per_shift": nurses_per_shift,
                    "shortage": round(max(0, monthly_demand - monthly_capacity), 1)
                })
            
            # -----------------------------
            # 4. FIXED SUPPLIES (static inventory)
            # -----------------------------
            elif r_type == "fixed":
                # Peak concurrent demand per shift
                peak_concurrent_need = avg_patients_per_shift * avg_use
                
                # Total available
                total_available = inventory['total_available']
                
                # Utilization
                utilization = peak_concurrent_need / total_available if total_available > 0 else 1
                
                status = ("OVER_CAPACITY" if utilization > 1 else
                         "HIGH_USAGE" if utilization > 0.8 else
                         "MODERATE" if utilization > 0.5 else
                         "OK")
                
                results.append({
                    "resource_name": resource_name,
                    "type": "fixed",
                    "unit": inventory['unit'],
                    "monthly_demand": round(peak_concurrent_need, 1),  # peak concurrent
                    "monthly_capacity": total_available,
                    "utilization": round(utilization * 100, 1),
                    "status": status,
                    "shortage": round(max(0, peak_concurrent_need - total_available), 1)
                })
        
        # Sort by status priority: OVER_CAPACITY → HIGH_USAGE → MODERATE → OK
        status_priority = {"OVER_CAPACITY": 0, "HIGH_USAGE": 1, "MODERATE": 2, "OK": 3}
        results.sort(key=lambda x: (status_priority.get(x["status"], 4), -x["utilization"]))
        
        return jsonify({
            "summary": {
                "predicted_monthly_visits": round(predicted_visits, 0),
                "adjusted_monthly_visits": round(adjusted_visits, 0),
                "general_factor": general_factor,
                "avg_patients_per_shift": round(avg_patients_per_shift, 1),
                "nurses_per_shift": nurses_per_shift,
                "shifts_per_month": shifts_per_month
            },
            "resources": results
        })
        
    except Exception as e:
        print(f"Error in /api/resource_needs: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/files')
def api_files():
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.xlsx')]
    return jsonify({'files': files})

@app.route('/api/download/<filename>')
def api_download(filename):
    safe_name = secure_filename(filename)
    file_path = os.path.join(INPUT_FOLDER, safe_name)
    if os.path.exists(file_path):
        return send_from_directory(INPUT_FOLDER, safe_name, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404



if __name__ == '__main__':
    app.run(debug=True)