import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import os

# --- Parameters ---
LOWCUT = 0.5       # Hz
HIGHCUT = 40.0     # Hz
ORDER = 4
FS = 1024          # Sampling frequency in Hz

# --- File paths ---
input_folder = r"C:\Users\aksha\OneDrive\Documents\007-Archith"  # Input folder containing Excel files
output_folder = r"C:\Users\aksha\OneDrive\Documents\nr1"  # Output directory

# --- Create output folder if it doesn't exist ---
os.makedirs(output_folder, exist_ok=True)

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

try:
    # Get all Excel files in the input folder
    excel_files = [f for f in os.listdir(input_folder) if f.endswith(('.xlsx', '.xls'))]
    print(f"Found {len(excel_files)} Excel files to process:")
    for file in excel_files:
        print(f"  - {file}")
    
    # Process each Excel file
    for excel_file in excel_files:
        input_path = os.path.join(input_folder, excel_file)
        output_filename = f'filtered_{excel_file}'
        output_path = os.path.join(output_folder, output_filename)
        
        print(f"\nProcessing file: {excel_file}")
        try:
            # Load Excel file and get sheet names
            excel_data = pd.ExcelFile(input_path)
            sheet_names = excel_data.sheet_names
            print(f"Found sheets: {sheet_names}")
            
            # Create an Excel writer object for this file
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Process each sheet
                for sheet_name in sheet_names:
                    print(f"\nProcessing sheet: {sheet_name}")
                    try:
                        df = pd.read_excel(input_path, sheet_name=sheet_name)
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

    print("\nAll files have been processed!")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print(f"Error type: {type(e).__name__}")
