import pandas as pd
import os

# --- CONFIGURATION ---

# 1. SETUP PATHS
# Get the folder where THIS script is located (.../main)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define where the RAW data is (.../synthetic_data_100_000)
raw_data_dir = os.path.abspath(os.path.join(script_dir, '..', 'synthetic_data_100_000'))

# Define where the CLEANED data will go (.../cleaned_data)
cleaned_data_dir = os.path.abspath(os.path.join(script_dir, '..', 'cleaned_data'))

# 2. DATA DEFINITIONS
datasets = {
    "person_data_100000.csv": [
        "gender", "migraine_days_per_month", "trigger_stress", "trigger_hormones", 
        "trigger_sleep", "trigger_weather", "trigger_medicine", "normal_sleep"
    ],
    "weather_data.csv": [
        "temp_mean", "wind_mean", "pressure_mean", "sun_irr_mean", 
        "sun_time_mean", "precip_total", "cloud_mean"
    ],
    "health_data_100000_365.csv": [
        "stress_intensity", "sleep_duration", "migraine_probability", 
        "p_stress", "p_hormones", "p_sleep"
    ]
}

def clean_and_save_data():
    print(f"--- Source Directory: {raw_data_dir}")
    print(f"--- Output Directory: {cleaned_data_dir}\n")

    # Verify source exists
    if not os.path.exists(raw_data_dir):
        print(f"[!] Error: Source directory not found: {raw_data_dir}")
        return

    # Create the 'cleaned_data' directory if it doesn't exist
    if not os.path.exists(cleaned_data_dir):
        os.makedirs(cleaned_data_dir)
        print(f"Created new directory: {cleaned_data_dir}")

    for filename, columns_to_keep in datasets.items():
        input_path = os.path.join(raw_data_dir, filename)
        output_path = os.path.join(cleaned_data_dir, f"cleaned_{filename}")

        print(f"Processing: {filename}...")

        if not os.path.exists(input_path):
            print(f"  [!] File not found: {input_path}")
            continue

        try:
            # STEP 1: Read
            df = pd.read_csv(input_path, usecols=columns_to_keep)

            # STEP 2: Clean
            initial_count = len(df)
            df.drop_duplicates(inplace=True)
            df.dropna(inplace=True)
            final_count = len(df)
            
            print(f"  - Dropped {initial_count - final_count} rows")
            print(f"  - Remaining: {final_count}")

            # STEP 3: Save to the NEW directory
            df.to_csv(output_path, index=False)
            print(f"  - Saved to: {output_path}\n")

        except ValueError as e:
            print(f"  [!] Column Error: {e}\n")
        except Exception as e:
            print(f"  [!] Error: {e}\n")

if __name__ == "__main__":
    clean_and_save_data()
    print("Done.")