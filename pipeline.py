# D:\tesFolder\pipeline.py

import os
import pandas as pd
import re
from openpyxl import load_workbook
import sys
import json
import numpy as np
import calendar
import warnings

def main():
    # Suppress specific UserWarnings from pandas
    warnings.simplefilter(action='ignore', category=UserWarning)

    # ----------------------------------------------------------------------
    # 0. Imports and Configuration
    # ----------------------------------------------------------------------
    print("="*50)
    print("PIPELINE: 0. CONFIGURATION AND SETUP")
    print("="*50)

    # ----------------------------
    # Paths
    # ----------------------------
    base_folder = r"D:\FlaskDev" 

    input_folder = os.path.join(base_folder, "data", "excel_folder")
    csv_folder = os.path.join(base_folder, "data", "csv_folder")
    out_folder = os.path.join(base_folder, "data", "processed")
    logs_folder = os.path.join(base_folder, "logs")

    # Specific File Paths
    master_csv = os.path.join(out_folder, "master_dataset.csv")
    cleaned_csv = os.path.join(out_folder, "master_dataset_cleaned.csv")
    case_dict_file = os.path.join(logs_folder, "case_dictionary.json")

    # Log files for each step
    log_file_01_excel_csv = os.path.join(logs_folder, "converted_files.txt")
    log_file_02_cleaning = os.path.join(logs_folder, "processed_files.log")
    log_file_03_master = os.path.join(logs_folder, "csv_master_log.txt")

    # ----------------------------
    # Create all folders (This will create them if they don't exist)
    # ----------------------------
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    print(f"Base folder set to: {base_folder}")
    print("Configuration and folders are ready.\n")


    # ----------------------------------------------------------------------
    # 1. Convert Excel to CSV (from 01_EXCEL_CSV.ipynb)
    # ----------------------------------------------------------------------
    print("="*50)
    print("PIPELINE: 1. EXCEL TO CSV CONVERSION")
    print("="*50)

    # Load log of already converted files
    if os.path.exists(log_file_01_excel_csv):
        with open(log_file_01_excel_csv, "r") as f:
            converted_files = f.read().splitlines()
    else:
        converted_files = []

    print(f"Found {len(converted_files)} files in Excel conversion log.")

    # Loop through all Excel files
    new_excel_files_converted = 0
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".xlsx") and file_name not in converted_files:
            try:
                file_path = os.path.join(input_folder, file_name)
                df = pd.read_excel(file_path)

                # Generate CSV file name
                csv_file_name = file_name.replace(".xlsx", ".csv")
                csv_path = os.path.join(csv_folder, csv_file_name)

                # Save to CSV
                df.to_csv(csv_path, index=False)
                print(f"Converted {file_name} to {csv_file_name}")

                # Add to log
                with open(log_file_01_excel_csv, "a") as f:
                    f.write(file_name + "\n")
                new_excel_files_converted += 1
            except Exception as e:
                print(f"❌ Error converting {file_name}: {e}")

    if new_excel_files_converted == 0:
        print("No new Excel files to convert. Already converted files were skipped.")
    else:
        print(f"Processing complete. Converted {new_excel_files_converted} new Excel files.")
    print("\n")


    # ----------------------------------------------------------------------
    # 2. Clean CSV Files (from 02_CSV_Cleaning.ipynb)
    # ----------------------------------------------------------------------
    print("="*50)
    print("PIPELINE: 2. INDIVIDUAL CSV CLEANING")
    print("="*50)

    # Final column schema
    final_columns = [
        "Month_year", "Consultation_Type", "Case",
        "Under 1 Male", "Under 1 Female",
        "1-4 Male", "1-4 Female",
        "5-9 Male", "5-9 Female",
        "10-14 Male", "10-14 Female",
        "15-18 Male", "15-18 Female",
        "19-24 Male", "19-24 Female",
        "25-29 Male", "25-29 Female",
        "30-34 Male", "30-34 Female",
        "35-39 Male", "35-39 Female",
        "40-44 Male", "40-44 Female",
        "45-49 Male", "45-49 Female",
        "50-54 Male", "50-54 Female",
        "55-59 Male", "55-59 Female",
        "60-64 Male", "60-64 Female",
        "65-69 Male", "65-69 Female",
        "70 Over Male", "70 Over Female"
    ]

    # Load already processed files
    processed_files_02 = set()
    if os.path.exists(log_file_02_cleaning):
        with open(log_file_02_cleaning, "r") as f:
            processed_files_02 = set(line.strip() for line in f.readlines())

    print(f"Found {len(processed_files_02)} files in CSV cleaning log.")

    # Scan and process only NEW CSV files
    new_csv_files_cleaned = 0
    for file in os.listdir(csv_folder):
        if file.endswith(".csv") and file not in processed_files_02:  # ✅ Skip logged files
            file_path = os.path.join(csv_folder, file)
            try:
                # STEP 1: Clean structure
                df = pd.read_csv(file_path)

                # Drop first column (extra index column)
                if df.columns[0].startswith("Unnamed: 0") or df.columns[0] == "":
                    df = df.drop(df.columns[0], axis=1)

                # Add 2 new columns on the left
                df.insert(0, "Month_year", "")
                df.insert(1, "Consultation_Type", "")

                # Trim/pad columns to match schema
                if df.shape[1] > len(final_columns):
                    df = df.iloc[:, :len(final_columns)]
                while df.shape[1] < len(final_columns):
                    df[f"Extra_{df.shape[1]}"] = ""

                # Rename columns
                df.columns = final_columns

                # STEP 2: Extract Month-Year from raw text
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                match = re.search(r"MONTH AND YEAR:\s*([A-Za-z]+)\s+(\d{4})", text)
                month_year_value = ""
                if match:
                    month_name = match.group(1).strip().title()
                    year = match.group(2).strip()
                    try:
                        month_num = list(calendar.month_name).index(month_name)
                        month_year_value = f"{year} - {month_num}"
                    except ValueError:
                        pass

                if month_year_value:
                    df["Month_year"] = month_year_value

                # STEP 3: Extract Consultation Type
                current_category = None
                found_categories = []

                for i, row in df.iterrows():
                    for cell in row.dropna().astype(str):
                        if "TOP 10" in cell.upper():
                            last_word = re.sub(r"[^\w]", "", cell.strip().split()[-1])
                            current_category = last_word.capitalize()
                            found_categories.append((i, current_category))
                            break

                    if current_category:
                        df.at[i, "Consultation_Type"] = current_category

                # STEP 4: Remove unwanted rows
                drop_indexes = []
                for i, row in df.iterrows():
                    for cell in row.dropna().astype(str):
                        if "PASIG CITY CHILDREN'S HOSPITAL/PASIG CITY COVID-19 REFERRAL CENTER" in cell.upper():
                            drop_indexes.extend(range(i, i + 9))
                            break

                for i, row in df.iterrows():
                    for cell in row.dropna().astype(str):
                        if "TOTAL" in cell.upper().strip():
                            drop_indexes.extend([i, i+1, i+2])
                            break

                drop_indexes = list(set(drop_indexes))
                df = df.drop(drop_indexes, errors="ignore").reset_index(drop=True)

                # STEP 5: Save and log
                df.to_csv(file_path, index=False)

                with open(log_file_02_cleaning, "a") as f:
                    f.write(file + "\n")

                print(f"✅ Processed and logged new file: {file}")
                new_csv_files_cleaned += 1

            except Exception as e:
                print(f"❌ Error processing {file}: {e}")

    if new_csv_files_cleaned == 0:
        print("No new CSV files to clean.")
    else:
        print(f"🎯 All {new_csv_files_cleaned} new CSV files processed and logged.")
    print("\n")


    # ----------------------------------------------------------------------
    # 3. Append CSVs to Master File (from 03_CSV_Masterfile.ipynb)
    # ----------------------------------------------------------------------
    print("="*50)
    print("PIPELINE: 3. APPEND CSVS TO MASTER FILE")
    print("="*50)

    # Load processed file log
    if os.path.exists(log_file_03_master):
        with open(log_file_03_master, "r") as f:
            processed_files_03 = set(line.strip() for line in f)
    else:
        processed_files_03 = set()

    print(f"Found {len(processed_files_03)} files in master file log.")

    # Scan for new CSV files
    csv_files = [f for f in os.listdir(csv_folder) if f.lower().endswith(".csv")]
    new_files_for_master = [f for f in csv_files if f not in processed_files_03]

    if not new_files_for_master:
        print("⚠️ No new files to append to master file.")
    else:
        print(f"📂 New files found to append: {new_files_for_master}")

        # Load or create case dictionary
        if os.path.exists(case_dict_file):
            with open(case_dict_file, "r") as f:
                case_dict = json.load(f)
        else:
            case_dict = {}
        print(f"Loaded case dictionary with {len(case_dict)} entries.")

        # Consultation mapping
        consultation_map = {
            "Consultation": 1,
            "Diagnosis": 2,
            "Mortality": 3
        }

        # Function to safely load CSV
        def safe_load_csv(file_path):
            return pd.read_csv(file_path, engine="python", on_bad_lines="skip")

        # Process new files
        processed_dfs = []
        master_columns = None
        
        if os.path.exists(master_csv):
            try:
                old_master_df = safe_load_csv(master_csv)
                master_columns = old_master_df.columns
            except pd.errors.EmptyDataError:
                print("Master CSV exists but is empty.")
                old_master_df = pd.DataFrame()
        else:
            old_master_df = pd.DataFrame()

        for file in new_files_for_master:
            file_path = os.path.join(csv_folder, file)
            try:
                df = safe_load_csv(file_path)

                if df.empty:
                    print(f"⚠️ Skipping {file}: no valid rows")
                    continue

                # Align columns with master if it exists
                if master_columns is not None:
                    for col in master_columns:
                        if col not in df.columns:
                            df[col] = None
                    df = df.reindex(columns=master_columns)
                
                # Clean 'Case' column
                if "Case" not in df.columns:
                    print(f"⚠️ Skipping {file}: no 'Case' column found")
                    continue

                df = df[df["Case"].notna()]
                df["Case"] = df["Case"].astype(str).str.strip()
                df = df[df["Case"] != ""]

                # Update case dictionary
                unique_cases = df["Case"].unique()
                for case in unique_cases:
                    if case not in case_dict:
                        case_dict[case] = len(case_dict) + 1

                df["Case"] = df["Case"].map(case_dict)

                # Encode Consultation_Type
                if "Consultation_Type" in df.columns:
                    df["Consultation_Type"] = df["Consultation_Type"].map(consultation_map)

                processed_dfs.append(df)
            except Exception as e:
                print(f"❌ Error processing {file} for master: {e}")

        # Append to master
        if processed_dfs:
            new_data = pd.concat(processed_dfs, ignore_index=True)

            if not old_master_df.empty:
                combined_df = pd.concat([old_master_df, new_data], ignore_index=True)
            else:
                combined_df = new_data
            
            combined_df.to_csv(master_csv, index=False)

            # Save case dictionary
            with open(case_dict_file, "w") as f:
                json.dump(case_dict, f, indent=4)

            # Update log
            with open(log_file_03_master, "a") as f:
                for file in new_files_for_master:
                    f.write(file + "\n")

            print(f"✅ Appended {len(new_files_for_master)} files to master")
            print(f"📊 Total rows in master: {len(combined_df)}")
            print(f"📖 Case dictionary size: {len(case_dict)}")
        else:
            print("⚠️ No valid rows to add from new files.")
    print("\n")


    # ----------------------------------------------------------------------
    # 4. Reshape Master to Long Format (from 04_Master_Long.ipynb)
    # ----------------------------------------------------------------------
    print("="*50)
    print("PIPELINE: 4. RESHAPE MASTER TO LONG FORMAT")
    print("="*50)

    # LOAD FILE
    try:
        final_df = pd.read_csv(master_csv)
        print(f"Loaded {master_csv} with {len(final_df)} rows.")
    except FileNotFoundError:
        print(f"Error: The file was not found at {master_csv}")
        print("Please make sure the 'master_dataset.csv' file exists before running this step.")
        return # Changed from sys.exit
    except pd.errors.EmptyDataError:
        print(f"Error: {master_csv} is empty. No data to process.")
        return # Changed from sys.exit

    # Strip spaces from all column headers
    final_df.columns = final_df.columns.str.strip()

    # CREATE MAPPING DICTIONARIES
    # Sex encoding
    sex_map = {"Male": 1, "Female": 0}

    # Age range encoding
    age_map = {
        "Under 1": 0, "1-4": 1, "5-9": 2, "10-14": 3, "15-18": 4, "19-24": 5,
        "25-29": 6, "30-34": 7, "35-39": 8, "40-44": 9, "45-49": 10, "50-54": 11,
        "55-59": 12, "60-64": 13, "65-69": 14, "70": 15, "70 Over": 15, "70 & OVER": 15
    }

    # Consultation_Type encoding
    consult_map = {name: idx for idx, name in enumerate(final_df["Consultation_Type"].dropna().unique(), start=1)}

    # Case encoding
    case_map = {name: idx for idx, name in enumerate(final_df["Case"].dropna().unique(), start=1)}

    # BUILD AGE+SEX MAPPING DICTIONARY
    mapping_dict = {}
    # Skip first 3 columns: Month_year, Consultation_Type, Case
    for col in final_df.columns[3:]:
        col_clean = col.strip()
        parts = col_clean.split()
        if len(parts) >= 2:
            sex = parts[-1]
            age = " ".join(parts[:-1])
            mapping_dict[col_clean] = {"Age_range": age, "Sex": sex}

    print("Mappings created.")

    # RESHAPE INTO LONG FORMAT
    reshaped_df = final_df.melt(
        id_vars=["Month_year", "Consultation_Type", "Case"],
        value_vars=final_df.columns[3:],
        var_name="Age_Sex",
        value_name="Total"
    )

    # Clean Age_Sex column
    reshaped_df["Age_Sex"] = reshaped_df["Age_Sex"].str.strip()

    # Map Age_range and Sex safely
    reshaped_df["Age_range"] = reshaped_df["Age_Sex"].map(
        lambda x: mapping_dict.get(x, {"Age_range": "Unknown"})["Age_range"]
    )
    reshaped_df["Sex"] = reshaped_df["Age_Sex"].map(
        lambda x: mapping_dict.get(x, {"Sex": "Unknown"})["Sex"]
    )

    print(f"Reshaped to long format. New row count: {len(reshaped_df)}")

    # SPLIT MONTH_YEAR INTO NUMERIC MONTH + YEAR
    reshaped_df["Month_year"] = pd.to_datetime(reshaped_df["Month_year"], errors="coerce")
    reshaped_df["Month"] = reshaped_df["Month_year"].dt.month
    reshaped_df["Year"] = reshaped_df["Month_year"].dt.year

    # HANDLE MISSING/EMPTY 'TOTAL' VALUES
    reshaped_df["Total"] = pd.to_numeric(reshaped_df["Total"], errors='coerce').fillna(0).astype(int)

    # ENCODE TO NUMERIC
    reshaped_df["Sex"] = reshaped_df["Sex"].map(sex_map).fillna(-1).astype(int)
    reshaped_df["Age_range"] = reshaped_df["Age_range"].map(age_map).fillna(-1).astype(int)
    reshaped_df["Consultation_Type"] = reshaped_df["Consultation_Type"].map(consult_map).fillna(-1).astype(int)
    reshaped_df["Case"] = reshaped_df["Case"].map(case_map).fillna(-1).astype(int)

    print("Date/Time split and columns encoded.")

    # FINAL NUMERIC STRUCTURE
    final_numeric_columns = ["Year", "Month", "Consultation_Type", "Case", "Sex", "Age_range", "Total"]
    reshaped_df = reshaped_df[final_numeric_columns]

    # --- THIS IS THE FIX ---
    # Drop exact duplicates from the final reshaped file
    original_rows = len(reshaped_df)
    reshaped_df.drop_duplicates(inplace=True)
    new_rows = len(reshaped_df)
    print(f"Deduplication complete: Removed {original_rows - new_rows} duplicate rows.")
    # -----------------------------------------------------------------

    # Drop rows where year or month could not be parsed
    reshaped_df.dropna(subset=['Year', "Month"], inplace=True)
    reshaped_df['Year'] = reshaped_df['Year'].astype(int)
    reshaped_df['Month'] = reshaped_df['Month'].astype(int)


    # SAVE TO CLEANED CSV
    reshaped_df.to_csv(cleaned_csv, index=False)

    print(f"✅ Cleaned numeric CSV saved as: {cleaned_csv}")
    print(f"Number of rows: {reshaped_df.shape[0]}, Number of columns: {reshaped_df.shape[1]}")

    # PRINT ENCODINGS
    print("\n🔑 Encodings Used:")
    print("Sex:", sex_map)
    print("Age_range:", age_map)
    print("Consultation_Type:", consult_map)
    # Print case_map in a more readable way if it's too large
    if len(case_map) > 20:
        print(f"Case: {{... {len(case_map)} entries ...}}")
    else:
        print("Case:", case_map)

    print("\n")
    print("="*50)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*50)

# This block ensures the code in main() only runs
# when you execute this file directly, not when it's imported.
if __name__ == "__main__":
    main()