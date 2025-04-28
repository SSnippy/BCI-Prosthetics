import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import os
import glob
from transform_analysis import bcitransform
import pywt
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import welch
from datetime import datetime
import keyboard

# --- Parameters ---
LOWCUT = 0.5       # Hz
HIGHCUT = 40.0     # Hz
ORDER = 4
FS = 1024          # Sampling frequency in Hz

# --- File paths ---
INPUT_BASE_FOLDER = r"Datapoints"  #add full path if error occurs
OUTPUT_BASE_FOLDER = r"nr"  #add full path if error occurs

print("Libraries imported successfully")

# --- Butterworth Bandpass Filter Functions ---
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    # Convert to numpy array and ensure it's float type
    data_array = np.array(data, dtype=float)
    # Remove any remaining NaN values
    data_array = data_array[~np.isnan(data_array)]
    if len(data_array) == 0:
        return np.zeros_like(data)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data_array)

# Create output base folder if it doesn't exist
os.makedirs(OUTPUT_BASE_FOLDER, exist_ok=True)

# Get all patient folders in the input directory
patient_folders = [f for f in os.listdir(INPUT_BASE_FOLDER) if os.path.isdir(os.path.join(INPUT_BASE_FOLDER, f))]
print(f"\nFound {len(patient_folders)} patient folders to process")

# Process each patient folder
for patient_folder in patient_folders:
    print(f"\n{'='*50}")
    print(f"Processing patient: {patient_folder}")
    print(f"{'='*50}")
    
    # Set up paths for this patient
    patient_input_path = os.path.join(INPUT_BASE_FOLDER, patient_folder)
    patient_output_path = os.path.join(OUTPUT_BASE_FOLDER, f"f_{patient_folder}")  # Changed from f"Transformed_{patient_folder}"
    os.makedirs(patient_output_path, exist_ok=True)
    
    # Get all Excel files in the patient folder
    excel_files = glob.glob(os.path.join(patient_input_path, "*.xlsx"))
    print(f"Found {len(excel_files)} Excel files to process")
    
    # Process each Excel file
    for excel_file in excel_files:
        try:
            print(f"\nProcessing file: {os.path.basename(excel_file)}")
            output_filename = f'f_{os.path.basename(excel_file)}'
            output_path = os.path.join(patient_output_path, output_filename)
            
            # Load Excel file and get sheet names
            excel_data = pd.ExcelFile(excel_file)
            sheet_names = excel_data.sheet_names
            print(f"Found sheets: {sheet_names}")
            
            # Create an Excel writer object for this file
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Process each sheet
                for sheet_name in sheet_names:
                    print(f"\nProcessing sheet: {sheet_name}")
                    try:
                        df = pd.read_excel(excel_file, sheet_name=sheet_name)
                        print(f"Successfully loaded data. Shape: {df.shape}")
                        print("\nColumns in the sheet:")
                        for i, col in enumerate(df.columns):
                            print(f"  {i+1}. {col}")
                        
                        # Clean the data - remove any rows with NaN values
                        df = df.dropna()
                        print(f"Data shape after removing NaN values: {df.shape}")
                        
                        # --- Filter the EEG columns ---
                        filtered_df = df.copy()
                        print("\nProcessing columns:")
                        
                        # Process all columns as EEG data
                        for col in df.columns:
                            print(f"Filtering column: {col}")
                            # Ensure the column is numeric
                            if not pd.api.types.is_numeric_dtype(df[col]):
                                print(f"  - Converting column {col} to numeric")
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                            # Apply the filter
                            try:
                                filtered_data = apply_bandpass_filter(df[col], LOWCUT, HIGHCUT, FS, ORDER)
                                filtered_df[col] = filtered_data
                                
                                print(f"  - Original data range: [{df[col].min():.2f}, {df[col].max():.2f}]")
                                print(f"  - Filtered data range: [{filtered_df[col].min():.2f}, {filtered_df[col].max():.2f}]")
                                print(f"  - NaN count in filtered data: {filtered_df[col].isna().sum()}")
                            except Exception as e:
                                print(f"  - Error filtering column {col}: {str(e)}")
                                filtered_df[col] = df[col]  # Keep original data if filtering fails

                        # Save the filtered data to a sheet in the output Excel file
                        filtered_df.to_excel(writer, sheet_name=f'Filtered_{sheet_name}', index=False)
                        print(f"Successfully saved filtered data for sheet {sheet_name}. Shape: {filtered_df.shape}")
                    
                    except Exception as e:
                        print(f"Error processing sheet {sheet_name}: {str(e)}")
                        print(f"Error type: {type(e).__name__}")
                        continue  # Continue with next sheet even if this one fails
            
            print(f"\nFinished processing file: {excel_file}")
            print(f"Output saved to: {output_path}")
        
        except Exception as e:
            print(f"Error processing file {excel_file}: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            continue  # Continue with next file even if this one fails
    
    print(f"\nCompleted processing patient: {patient_folder}")
    print(f"Results saved in: {patient_output_path}")

print("\nAll processing complete!")
print(f"All filtered data is saved in: {OUTPUT_BASE_FOLDER}")
