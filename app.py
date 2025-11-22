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
ALLOWED_EXTENSIONS = {'xlsx'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = INPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'your-secret-key-goes-here' 

# --- RESOURCE CONFIG ---
SHIFTS_PER_DAY = 2  # adjust if you use 2 shifts/day

NURSES_PER_SHIFT = 7        # you said 7 nurses per shift on average

# Per-shift capacities & counts
RESOURCE_INVENTORY = {
    # ======================
    # CONSUMABLES – PER SHIFT
    # ======================
    "syringe_small": {  # 1cc / 3cc / 5cc / 10cc
        "display_name": "Syringes 1/3/5/10cc",
        "unit": "pcs",
        "capacity_per_shift": 100,    # 100 per shift
        "type": "consumable"
    },
    "syringe_large": {  # 20cc / 50cc
        "display_name": "Syringes 20/50cc",
        "unit": "pcs",
        "capacity_per_shift": 20,     # 20 per shift
        "type": "consumable"
    },
    "macroset": {
        "display_name": "Macroset",
        "unit": "sets",
        "capacity_per_shift": 50,     # 50 per shift
        "type": "consumable"
    },
    "microset": {
        "display_name": "Microset",
        "unit": "sets",
        "capacity_per_shift": 50,     # 50 per shift
        "type": "consumable"
    },
    "oxygen_cannula_mask": {
        "display_name": "Oxygen cannula/mask",
        "unit": "pcs",
        "capacity_per_shift": 50,     # 50 per shift
        "type": "consumable"
    },
    "soluset": {
        "display_name": "Soluset",
        "unit": "sets",
        "capacity_per_shift": 50,     # 50 per shift
        "type": "consumable"
    },
    "nebulizer_kit": {
        "display_name": "Nebulizer kit",
        "unit": "kits",
        "capacity_per_shift": 50,     # 50 per shift
        "type": "consumable"
    },
    "iv_cannula_20g": {
        "display_name": "IV cannula G20",
        "unit": "pcs",
        "capacity_per_shift": 100,    # 100 per shift
        "type": "consumable"
    },
    "iv_cannula_22g": {
        "display_name": "IV cannula G22",
        "unit": "pcs",
        "capacity_per_shift": 100,    # 100 per shift
        "type": "consumable"
    },
    "iv_cannula_24g": {
        "display_name": "IV cannula G24",
        "unit": "pcs",
        "capacity_per_shift": 100,    # 100 per shift
        "type": "consumable"
    },
    "iv_cannula_26g": {
        "display_name": "IV cannula G26",
        "unit": "pcs",
        "capacity_per_shift": 100,    # 100 per shift
        "type": "consumable"
    },
    "hgt_strips": {
        "display_name": "HGT strips",
        "unit": "strips",
        "capacity_per_shift": 100,    # 100 per shift
        "type": "consumable"
    },
    "sterile_gloves": {
        "display_name": "Sterile gloves (pairs, all sizes)",
        "unit": "pairs",
        "capacity_per_shift": 10 * 4,  # 10 per size; assume 4 sizes ≈ 40 pairs/shift
        "type": "consumable"
    },

    # ======================
    # CONSUMABLES – PER NURSE
    # ======================
    "n95_mask": {
        "display_name": "N95 mask",
        "unit": "pcs",
        # 1 per nurse per shift → 7 nurses ≈ 7 per shift
        "capacity_per_shift": NURSES_PER_SHIFT * 1,
        "type": "consumable_per_nurse"
    },
    "gown": {
        "display_name": "Gown",
        "unit": "pcs",
        # 1 per nurse per shift → 7 nurses ≈ 7 per shift
        "capacity_per_shift": NURSES_PER_SHIFT * 1,
        "type": "consumable_per_nurse"
    },

    # ======================
    # FIXED SUPPLIES (TREATED AS EQUIPMENT CAPACITY)
    # ======================
    "stretcher": {
        "display_name": "Stretchers",
        "unit": "units",
        "capacity_per_shift": 15,     # total available
        "type": "equipment"
    },
    "crib": {
        "display_name": "Crib",
        "unit": "units",
        "capacity_per_shift": 1,
        "type": "equipment"
    },
    "wheelchair": {
        "display_name": "Wheelchair",
        "unit": "units",
        "capacity_per_shift": 4,
        "type": "equipment"
    },
    "bassinet": {
        "display_name": "Bassinet",
        "unit": "units",
        "capacity_per_shift": 2,
        "type": "equipment"
    },

    # ======================
    # EQUIPMENT (CAPACITY-BASED, NOT CONSUMED)
    # ======================
    "cardiac_monitor_wall": {
        "display_name": "Cardiac monitor (wall)",
        "unit": "units",
        "capacity_per_shift": 8,
        "type": "equipment"
    },
    "cardiac_monitor_portable": {
        "display_name": "Cardiac monitor (portable)",
        "unit": "units",
        "capacity_per_shift": 7,
        "type": "equipment"
    },
    "suction_wall": {
        "display_name": "Wall suction",
        "unit": "units",
        "capacity_per_shift": 8,
        "type": "equipment"
    },
    "suction_portable": {
        "display_name": "Portable suction",
        "unit": "units",
        "capacity_per_shift": 2,
        "type": "equipment"
    },
    "ecg_machine": {
        "display_name": "ECG machine",
        "unit": "units",
        "capacity_per_shift": 1,
        "type": "equipment"
    },
    "nebulizer_machine": {
        "display_name": "Nebulizer machine",
        "unit": "units",
        "capacity_per_shift": 4,
        "type": "equipment"
    },
    "bp_apparatus": {
        "display_name": "BP apparatus (standalone)",
        "unit": "units",
        "capacity_per_shift": 2,
        "type": "equipment"
    },
    "pulse_oximeter_portable": {
        "display_name": "Pulse oximeter (portable)",
        "unit": "units",
        "capacity_per_shift": 2,
        "type": "equipment"
    },
    "thermometer": {
        "display_name": "Thermometer",
        "unit": "units",
        "capacity_per_shift": 5,
        "type": "equipment"
    },
    "defibrillator": {
        "display_name": "Defibrillator",
        "unit": "units",
        "capacity_per_shift": 2,
        "type": "equipment"
    },
    "oxygen_tank": {
        "display_name": "O₂ tank with gauge",
        "unit": "units",
        "capacity_per_shift": 15,
        "type": "equipment"
    },
    "infusion_pump": {
        "display_name": "Infusion pump",
        "unit": "units",
        "capacity_per_shift": 5,
        "type": "equipment"
    },
    "spine_board": {
        "display_name": "Spine board",
        "unit": "units",
        "capacity_per_shift": 1,
        "type": "equipment"
    },
    "cervical_collar": {
        "display_name": "Cervical collar",
        "unit": "units",
        "capacity_per_shift": 2,
        "type": "equipment"
    },
    "glucose_monitor": {
        "display_name": "Glucose monitor",
        "unit": "units",
        "capacity_per_shift": 2,
        "type": "equipment"
    },
    "crash_cart": {
        "display_name": "Crash cart",
        "unit": "units",
        "capacity_per_shift": 2,
        "type": "equipment"
    },
    "mayo_table": {
        "display_name": "Mayo table",
        "unit": "units",
        "capacity_per_shift": 2,
        "type": "equipment"
    },
    "ambu_bag_mask": {
        "display_name": "Ambu bag mask",
        "unit": "units",
        "capacity_per_shift": 2,
        "type": "equipment"
    },
    "intubation_set": {
        "display_name": "Intubation set",
        "unit": "sets",
        "capacity_per_shift": 4,
        "type": "equipment"
    },
}

# Average resource use per patient for key diagnoses
# Keys MUST match the case names in your case_dictionary.json

CASE_RESOURCE_USAGE = {
    # --- Simple OPD-style cases ---
    "FEVER": {
        "sterile_gloves": 0.5,
        "hgt_strips": 0.3,
        "thermometer": 0.05
    },
    "COUGH": {
        "sterile_gloves": 0.5,
        "nebulizer_kit": 0.2,
        "nebulizer_machine": 0.1,
        "oxygen_cannula_mask": 0.05
    },
    "COUGH AND COLDS": {
        "sterile_gloves": 0.5,
        "nebulizer_kit": 0.2,
        "nebulizer_machine": 0.1
    },
    "URTI": {
        "sterile_gloves": 0.5,
        "nebulizer_kit": 0.2,
        "nebulizer_machine": 0.1
    },

    # --- GI / dehydration cases ---
    "VOMITING": {
        "sterile_gloves": 1.0,
        "iv_cannula_22g": 0.5,
        "macroset": 0.3,
        "microset": 0.3,
        "hgt_strips": 0.5
    },
    "LBM": {
        "sterile_gloves": 1.0,
        "iv_cannula_22g": 0.5,
        "macroset": 0.3,
        "microset": 0.3,
        "hgt_strips": 0.5
    },
    "AGE": {  # your case 11
        "sterile_gloves": 1.0,
        "iv_cannula_22g": 0.8,
        "macroset": 0.5,
        "microset": 0.5,
        "soluset": 0.5,
        "hgt_strips": 0.8
    },
    "ACUTE GASTROENTERITIS": {  # 46
        "sterile_gloves": 1.0,
        "iv_cannula_22g": 0.8,
        "macroset": 0.5,
        "microset": 0.5,
        "soluset": 0.5,
        "hgt_strips": 0.8
    },
    "ABDOMINAL PAIN": {
        "sterile_gloves": 1.0,
        "hgt_strips": 0.3
    },
    "ACUTE GASTRITIS": {
        "sterile_gloves": 1.0,
        "iv_cannula_22g": 0.4,
        "hgt_strips": 0.3
    },

    # --- UTI ---
    "UTI": {
        "sterile_gloves": 1.0,
        "iv_cannula_22g": 0.5,
        "hgt_strips": 0.4
    },
    "URINARY TRACT INFECTION": {
        "sterile_gloves": 1.0,
        "iv_cannula_22g": 0.5,
        "hgt_strips": 0.4
    },

    # --- Respiratory / pneumonia / SOB ---
    "SOB": {
        "sterile_gloves": 1.0,
        "nebulizer_kit": 0.5,
        "nebulizer_machine": 0.3,
        "oxygen_cannula_mask": 0.6,
        "cardiac_monitor_portable": 0.3,
        "pulse_oximeter_portable": 0.4
    },
    "PNEUMONIA": {
        "sterile_gloves": 1.0,
        "nebulizer_kit": 0.7,
        "nebulizer_machine": 0.4,
        "oxygen_cannula_mask": 0.8,
        "cardiac_monitor_wall": 0.4,
        "pulse_oximeter_portable": 0.4,
        "hgt_strips": 0.5
    },
    "CAP": {
        "sterile_gloves": 1.0,
        "nebulizer_kit": 0.5,
        "nebulizer_machine": 0.3,
        "oxygen_cannula_mask": 0.6,
        "cardiac_monitor_wall": 0.3
    },
    "CAP HIGH RISK": {
        "sterile_gloves": 1.0,
        "nebulizer_kit": 0.8,
        "nebulizer_machine": 0.5,
        "oxygen_cannula_mask": 0.9,
        "cardiac_monitor_wall": 0.5,
        "infusion_pump": 0.3
    },

    # --- Dengue / febrile illnesses with fluids ---
    "DENGUE FEVER": {
        "sterile_gloves": 1.0,
        "iv_cannula_22g": 1.0,
        "macroset": 0.6,
        "microset": 0.6,
        "soluset": 0.6,
        "hgt_strips": 0.8
    },

    # --- Trauma / injury ---
    "LACERATED WOUND": {
        "sterile_gloves": 1.5,
        "elastic_bandage": 0.5
    },
    "LACERATION": {
        "sterile_gloves": 1.5,
        "elastic_bandage": 0.5
    },
    "FALL": {
        "sterile_gloves": 1.0,
        "stretcher": 0.2,
        "wheelchair": 0.2,
        "cervical_collar": 0.1,
        "spine_board": 0.05
    },
    "FRACTURE": {
        "sterile_gloves": 1.0,
        "elastic_bandage": 0.8,
        "stretcher": 0.3,
        "wheelchair": 0.3
    },
    "CONTUSION": {
        "sterile_gloves": 0.5,
        "elastic_bandage": 0.5
    },
    "ABRASION": {
        "sterile_gloves": 0.5,
        "elastic_bandage": 0.5
    },

    # --- High-acuity / ER resuscitation style (simplified) ---
    "CARDIAC ARREST": {
        "sterile_gloves": 2.0,
        "ambu_bag_mask": 0.5,
        "intubation_set": 0.3,
        "crash_cart": 0.5,
        "defibrillator": 0.3,
        "oxygen_tank": 0.8
    },
    "OUT OF HOSPITAL CARDIAC ARREST": {
        "sterile_gloves": 2.0,
        "ambu_bag_mask": 0.5,
        "intubation_set": 0.3,
        "crash_cart": 0.5,
        "defibrillator": 0.3,
        "oxygen_tank": 0.8
    },
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
    try:
        model = joblib.load(MODEL_FILE)
        data = request.json
        X_new = pd.DataFrame({
            "Year": [int(data['year'])],
            "Month": [int(data['month'])],
            "Consultation_Type": [int(data['consult_type'])],
            "Case": [int(data['case_id'])],
            "Sex": [int(data['sex'])],
            "Age_range": [int(data['age_range'])]
        })
        prediction = model.predict(X_new)
        return jsonify({'prediction': round(prediction[0], 2)})
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
    using predictions from the trained Random Forest model.
    """
    try:
        month = int(request.json['month'])
        # --- 1. Load data and case dictionary ---
        df = pd.read_csv(CLEANED_CSV)
        with open(CASE_DICT_FILE, 'r') as f:
            case_dict = json.load(f)
        inv_case_dict_numeric = {v: k for k, v in case_dict.items()}

        # --- 2. Check if model file exists ---
        if not os.path.exists(MODEL_FILE):
            return jsonify({'error': 'Model file not found. Please train the model first.'}), 500

        # --- 3. Load trained model ---
        model = joblib.load(MODEL_FILE)

        # --- 4. Filter by selected month (and create a safe copy) ---
        df_month_for_check = df[df['Month'] == month]
        if df_month_for_check.empty:
            return jsonify({'error': f'No historical data found for {pd.to_datetime(f"2024-{month}-01").strftime("%B")}. Cannot make predictions.'}), 404
        df_month = df_month_for_check.copy()


        # --- 5. Prepare features for prediction ---
        features = ["Year", "Month", "Consultation_Type", "Case", "Sex", "Age_range"]
        
        # Ensure all feature columns are present, even if empty
        for col in features:
            if col not in df_month.columns:
                return jsonify({'error': f'Missing required column: {col}'}), 500
                
        X_month = df_month[features]

        # --- 6. Predict totals using the trained model ---
        df_month["Predicted_Total"] = model.predict(X_month)

        # --- 7. Compare actual vs predicted totals ---
        actual_total = df_month["Total"].sum()
        predicted_total = df_month["Predicted_Total"].sum()

        # --- 8. Evaluate monthly accuracy ---
        r2 = r2_score(df_month["Total"], df_month["Predicted_Total"])
        mae = mean_absolute_error(df_month["Total"], df_month["Predicted_Total"])
        rmse = np.sqrt(mean_squared_error(df_month["Total"], df_month["Predicted_Total"]))

        # --- 9. Get categorized top 10 cases BASED ON PREDICTION ---
        consultation_data = get_top_10_by_type(df_month, 1, inv_case_dict_numeric, column_to_sum='Predicted_Total')
        diagnosis_data = get_top_10_by_type(df_month, 2, inv_case_dict_numeric, column_to_sum='Predicted_Total')
        mortality_data = get_top_10_by_type(df_month, 3, inv_case_dict_numeric, column_to_sum='Predicted_Total')

        # --- 10. Return JSON response ---
        return jsonify({
            'month': month,
            'total_summary': {
                'actual_total': round(float(actual_total), 2),
                'predicted_total': round(float(predicted_total), 2),
                'accuracy_metrics': {
                    'R²': round(r2, 4),
                    'MAE': round(mae, 4),
                    'RMSE': round(rmse, 4)
                }
            },
            'consultation': consultation_data,
            'diagnosis': diagnosis_data,
            'mortality': mortality_data
        })

    except FileNotFoundError:
        return jsonify({'error': 'Data or case dictionary file not found. Please process files.'}), 404
    except Exception as e:
        print(f"Error in /api/top_cases: {e}")
        return jsonify({'error': str(e)}), 500
    

@app.route('/api/resource_needs', methods=['POST'])
def api_resource_needs():
    """
    Compute predicted resource needs for the selected month, based on
    predicted top 10 cases and per-patient resource usage.
    """
    try:
        month = int(request.json['month'])

        # --- 1. Load data & case dictionary ---
        df = pd.read_csv(CLEANED_CSV)
        with open(CASE_DICT_FILE, 'r') as f:
            case_dict = json.load(f)
        inv_case_dict_numeric = {v: k for k, v in case_dict.items()}

        # --- 2. Load model ---
        if not os.path.exists(MODEL_FILE):
            return jsonify({'error': 'Model file not found. Please train the model first.'}), 500
        model = joblib.load(MODEL_FILE)

        # --- 3. Filter selected month ---
        df_month = df[df['Month'] == month].copy()
        if df_month.empty:
            return jsonify({'error': 'No historical data for this month. Cannot compute resource needs.'}), 404

        # --- 4. Predict totals for each row ---
        features = ["Year", "Month", "Consultation_Type", "Case", "Sex", "Age_range"]
        for col in features:
            if col not in df_month.columns:
                return jsonify({'error': f'Missing required column: {col}'}), 500

        df_month["Predicted_Total"] = model.predict(df_month[features])

        # --- 5. Aggregate predicted patients per Case (all consultation types combined) ---
        agg_cases = (
            df_month.groupby("Case")["Predicted_Total"]
                    .sum()
                    .reset_index()
                    .sort_values("Predicted_Total", ascending=False)
        )

        # Top 10 cases by predicted count
        top10_cases = agg_cases.head(10).copy()
        top10_cases["CaseName"] = top10_cases["Case"].map(inv_case_dict_numeric).fillna("Unknown Case")

        # --- 6. Compute monthly resource demand from these top cases ---
        resource_demand = {res_key: 0.0 for res_key in RESOURCE_INVENTORY.keys()}
        case_details = []

        for _, row in top10_cases.iterrows():
            case_name = row["CaseName"]
            predicted_patients = float(row["Predicted_Total"])

            usage_map = CASE_RESOURCE_USAGE.get(case_name, {})
            case_details.append({
                "CaseName": case_name,
                "Predicted_Patients": round(predicted_patients, 2),
                "resources_used": usage_map
            })

            for res_key, units_per_patient in usage_map.items():
                if res_key not in resource_demand:
                    resource_demand[res_key] = 0.0
                resource_demand[res_key] += predicted_patients * units_per_patient

        # --- 7. Convert per-shift capacity to monthly capacity, evaluate utilization ---
        days_in_month = 30  # or compute from Year/Month if you like
        total_shifts = days_in_month * SHIFTS_PER_DAY

        resource_summary = []
        for res_key, demand in resource_demand.items():
            info = RESOURCE_INVENTORY.get(res_key, {
                "display_name": res_key,
                "unit": "units",
                "capacity_per_shift": 0,
                "type": "unknown"
            })

            cap_per_shift = info.get("capacity_per_shift", 0)
            monthly_capacity = cap_per_shift * total_shifts if cap_per_shift else 0

            utilization = (demand / monthly_capacity) if monthly_capacity > 0 else None

            if utilization is None:
                continue
            elif utilization < 0.7:
                status = "OK"
            elif utilization < 1.0:
                status = "HIGH_USAGE"
            else:
                status = "OVER_CAPACITY"

            resource_summary.append({
                "resource_key": res_key,
                "resource_name": info.get("display_name", res_key),
                "unit": info.get("unit", "units"),
                "type": info.get("type", "unknown"),
                "predicted_monthly_demand": round(demand, 2),
                "monthly_capacity": round(monthly_capacity, 2),
                "utilization": round(utilization, 2) if utilization is not None else None,
                "status": status
            })

        # Sort so critical ones (high or over capacity) appear first
        resource_summary.sort(
            key=lambda x: (x["status"] in ["OVER_CAPACITY", "HIGH_USAGE"], x["utilization"] or 0),
            reverse=True
        )

        return jsonify({
            "month": month,
            "top_cases": case_details,
            "resources": resource_summary
        })

    except FileNotFoundError:
        return jsonify({'error': 'Data or dictionary file not found. Please process files.'}), 404
    except Exception as e:
        print(f"Error in /api/resource_needs: {e}")
        return jsonify({'error': str(e)}), 500

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