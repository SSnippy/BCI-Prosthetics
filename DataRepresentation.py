import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Global constants
SAMPLING_FREQ = 1024  # Sampling frequency (Hz)
TIME_WINDOW = 0.1  # Time window in seconds
TIME_TICK_INTERVAL = 0.01  # Time tick interval in seconds

print("Libraries imported successfully")

# Define the folder containing all movement files
MOVEMENTS_FOLDER = "C:/Users/aksha/OneDrive/Documents/nr1"  
OUTPUT_BASE_FOLDER = r"C:/Users/aksha/OneDrive/Documents/output"

# Get all Excel files in the movements folder
movement_files = glob.glob(os.path.join(MOVEMENTS_FOLDER, "*.xlsx"))
print(f"\nFound {len(movement_files)} movement files to process")

# Process each movement file
for eeg_data_path in movement_files:
    try:
        print(f"\n{'='*100}")
        print(f"Processing file: {os.path.basename(eeg_data_path)}")
        print(f"{'='*100}")
        
        # Create parent folder with same name as input file (without extension)
        input_filename = os.path.splitext(os.path.basename(eeg_data_path))[0]
        PARENT_FOLDER = os.path.join(OUTPUT_BASE_FOLDER, input_filename)
        os.makedirs(PARENT_FOLDER, exist_ok=True)

        # Create plots subfolder within the parent folder
        PLOTS_FOLDER = os.path.join(PARENT_FOLDER, "plots")
        os.makedirs(PLOTS_FOLDER, exist_ok=True)

        print(f"Output will be saved in: {PARENT_FOLDER}")
        print(f"Plots will be saved in: {PLOTS_FOLDER}")

        def extract_window(eeg_signal, start_time):
            """
            Extract a 0.2-second window starting from the specified time.
            
            Args:
                eeg_signal: The complete EEG signal
                start_time: Start time in seconds
                
            Returns:
                window_signal: The extracted signal window
                window_time: Time array for the window
            """
            # Calculate start and end indices
            start_idx = int(start_time * SAMPLING_FREQ)
            end_idx = int((start_time + TIME_WINDOW) * SAMPLING_FREQ)
            
            # Ensure indices are within bounds
            start_idx = max(0, min(start_idx, len(eeg_signal)))
            end_idx = max(0, min(end_idx, len(eeg_signal)))
            
            # Extract the window
            window_signal = eeg_signal[start_idx:end_idx]
            window_time = np.linspace(start_time, start_time + TIME_WINDOW, len(window_signal))
            
            return window_signal, window_time

        def process_electrode(eeg_signal, electrode_name):
            """
            Process and plot the EEG signal for a single electrode.
            
            Args:
                eeg_signal: The EEG signal array
                electrode_name: Name of the electrode
                
            Returns:
                bool: True if successful, False otherwise
            """
            try:
                # Remove NaN values
                eeg_signal = eeg_signal[~np.isnan(eeg_signal)]
                
                # Calculate total duration
                total_duration = len(eeg_signal) / SAMPLING_FREQ
                num_windows = int(np.ceil(total_duration / TIME_WINDOW))
                
                print(f"\nProcessing {electrode_name}:")
                print(f"Total duration: {total_duration:.2f} seconds")
                print(f"Number of samples: {len(eeg_signal)}")
                print(f"Number of windows to create: {num_windows}")
                
                # Create electrode-specific folder
                electrode_folder = os.path.join(PLOTS_FOLDER, electrode_name.replace(' ', '_'))
                os.makedirs(electrode_folder, exist_ok=True)
                
                # Process each time window
                for window_idx in range(num_windows):
                    # Calculate start and end time for this window
                    start_time = window_idx * TIME_WINDOW
                    end_time = min(start_time + TIME_WINDOW, total_duration)
                    
                    print(f"Creating window {window_idx + 1}/{num_windows}: {start_time:.1f}s - {end_time:.1f}s")
                    
                    # Calculate start and end indices
                    start_idx = int(start_time * SAMPLING_FREQ)
                    end_idx = int(end_time * SAMPLING_FREQ)
                    
                    # Get the signal segment for this window
                    window_signal = eeg_signal[start_idx:end_idx]
                    
                    # Create time array for this window
                    window_time = np.linspace(start_time, end_time, len(window_signal))
                    
                    # Create plot for this window
                    plt.figure(figsize=(8, 4))  # Slightly smaller width for 0.1s window
                    plt.plot(window_time, window_signal, label='EEG Signal', color='purple', linewidth=1)
                    
                    # Create title with time duration information
                    title = f'EEG Signal for {electrode_name}\n'
                    title += f'Time: {start_time:.2f}s - {end_time:.2f}s (Window {window_idx + 1}/{num_windows})'
                    plt.title(title, fontsize=12)
                    
                    plt.xlabel('Time (seconds)', fontsize=10)
                    plt.ylabel('Amplitude (µV)', fontsize=10)
                    
                    # Set x-axis ticks every 0.01 seconds
                    plt.xticks(np.arange(start_time, end_time + TIME_TICK_INTERVAL, TIME_TICK_INTERVAL))
                    plt.yticks(fontsize=8)
                    
                    plt.grid(True, which='both', linestyle='--', alpha=0.5)
                    plt.legend(fontsize=10)
                    
                    # Set y-axis limits with some padding
                    window_y_min = np.min(window_signal) - 0.1 * np.abs(np.min(window_signal))
                    window_y_max = np.max(window_signal) + 0.1 * np.abs(np.max(window_signal))
                    plt.ylim(window_y_min, window_y_max)
                    
                    plt.tight_layout()
                    
                    # Save the plot in electrode-specific folder
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    plot_filename = f"window_{window_idx + 1}_{timestamp}.png"
                    plot_filepath = os.path.join(electrode_folder, plot_filename)
                    try:
                        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
                        print(f"Saved plot for {electrode_name} window {window_idx + 1} to: {plot_filepath}")
                    except Exception as e:
                        print(f"Error saving plot for window {window_idx + 1}: {str(e)}")
                        # Try to save with lower DPI
                        try:
                            plt.savefig(plot_filepath, dpi=150, bbox_inches='tight')
                            print(f"Saved plot with reduced DPI (150)")
                        except Exception as e2:
                            print(f"Failed to save plot even with reduced DPI: {str(e2)}")
                    
                    plt.close()
                    plt.clf()  # Clear the figure to free memory
                
                print(f"\nCompleted processing {electrode_name}")
                print(f"Total windows created: {num_windows}")
                return True
            except Exception as e:
                print(f"Error processing electrode {electrode_name}: {str(e)}")
                return False

        # Get list of all sheets in the Excel file
        excel_file = pd.ExcelFile(eeg_data_path)
        sheet_names = excel_file.sheet_names
        print(f"\nFound {len(sheet_names)} sheets in the Excel file:")
        for sheet in sheet_names:
            print(f"- {sheet}")
        
        # Read and combine all sheets
        all_data = []
        for sheet in sheet_names:
            try:
                df = pd.read_excel(eeg_data_path, sheet_name=sheet)
                all_data.append(df)
            except Exception as e:
                print(f"Error reading sheet {sheet}: {str(e)}")
        
        # Concatenate all sheets vertically
        eeg_df = pd.concat(all_data, ignore_index=True)
        print("\nExcel file loaded successfully")
        print(f"Data shape after combining all sheets: {eeg_df.shape}")
        print(f"Total duration: {len(eeg_df)/SAMPLING_FREQ:.2f} seconds")
        
        # Print available electrodes
        print("\nProcessing the following electrodes:")
        for col in eeg_df.columns:
            print(f"- {col}")

        # Process time domain plots
        print("\nProcessing time domain plots...")
        for electrode in eeg_df.columns:
            if electrode != 'EKG-REF':  # Skip the EKG reference channel
                print(f"Processing time domain plots for electrode: {electrode}")
                eeg_signal = eeg_df[electrode].values
                process_electrode(eeg_signal, electrode)
        
        print("\nAnalysis complete. All plots have been saved.")
        print(f"Results are saved in: {PARENT_FOLDER}")
        print(f"Time domain plots are in: {PLOTS_FOLDER}")
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        continue 