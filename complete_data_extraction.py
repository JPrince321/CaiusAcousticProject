import pandas as pd
import numpy as np
import re
import os
import glob

def process_rt60_raw(directory_path, output_csv):
    """Parses REW txt files and converts them directly into a wide-format CSV."""
    all_records = []
    
    # Define known locations to search for in the measurement string
    KNOWN_LOCATIONS = ['SvMv', 'ShMv', 'ShM45', 'SvM45']
    
    # Read all text files in the target directory
# Read all text files in the target directory AND its subfolders
    file_pattern = os.path.join(directory_path, "**", "*.txt")
    files = glob.glob(file_pattern, recursive=True)
    
    if not files:
        print(f"No .txt files found in the specified directory: {directory_path}")
        return
        
    for filepath in files:
        filename = os.path.basename(filepath)
        
        # --- Extract Condition directly from the file name ---
        # e.g., "RT60_Medium Loose.txt" -> "Medium Loose"
        condition = filename.replace("RT60_", "").replace(".txt", "")
        
        current_location = "Unknown"
        current_repeat = "1"
        
        with open(filepath, 'r') as file:
            for line in file:
                line = line.strip()
                
                # Extract Location and Repeat
                if line.startswith("Measurement:"):
                    # Extract the repeat number from the end of the line (requiring a space before it)
                    match = re.search(r'Measurement:\s*(.*?)\s*(\d+)$', line)
                    if match:
                        measurement_text = match.group(1).strip()
                        current_repeat = match.group(2).strip()
                    else:
                        measurement_text = line.replace("Measurement:", "").strip()
                        current_repeat = "1"
                        
                    # Extract Location from the isolated measurement string
                    for loc in KNOWN_LOCATIONS:
                        if loc.lower() in measurement_text.lower():
                            current_location = loc 
                            break
                    continue
                
                # Extract data rows
                if re.match(r'^(\d+|full)\s+1/3', line):
                    parts = line.split()
                    
                    def to_float(val):
                        try:
                            return float(val)
                        except ValueError:
                            return np.nan

                    record = {
                        'File': filename,
                        'Condition': condition,       # Pulled from filename
                        'Location': current_location, # Pulled from Measurement string
                        'Repeat': current_repeat,
                        'Freq': parts[0],
                        'EDT': to_float(parts[2]),
                        'T20': to_float(parts[4]),
                        'T30': to_float(parts[6]),
                        'Topt': to_float(parts[8]),
                        'T60M': to_float(parts[12]),
                        'C50': to_float(parts[-4]),
                        'C80': to_float(parts[-3]),
                        'D50': to_float(parts[-2]),
                        'TS': to_float(parts[-1])
                    }
                    all_records.append(record)

    df = pd.DataFrame(all_records)
    
    if df.empty:
        print("No valid data found in the text files.")
        return

    # Melt to Long format
    df_melted = df.melt(
        id_vars=['File', 'Condition', 'Location', 'Repeat', 'Freq'], 
        value_vars=['EDT', 'T20', 'T30', 'Topt', 'T60M', 'C50', 'C80', 'D50', 'TS'],
        var_name='Metric', 
        value_name='Value'
    )
    
    # Pivot to Wide format
    df_pivot = df_melted.pivot_table(
        index=['File', 'Condition', 'Location', 'Repeat', 'Metric'], 
        columns='Freq', 
        values='Value'
    ).reset_index()
    
    # Sort frequency columns properly (numbers first, then 'full')
    freq_cols = [c for c in df_pivot.columns if c not in ['File', 'Condition', 'Location', 'Repeat', 'Metric', 'full']]
    freq_cols.sort(key=float) 
    if 'full' in df_pivot.columns:
        freq_cols.append('full')
        
    final_columns = ['File', 'Condition', 'Location', 'Repeat', 'Metric'] + freq_cols
    df_final = df_pivot[final_columns]
    
    df_final.to_csv(output_csv, index=False)
    print(f"\nSuccess! Processed data exported to {output_csv}")

# --- EXECUTION ---
directory_to_search = "C:\\Users\\joshu\\Documents\\Cambridge\\4th Year Work\\Project\\Panel Measurements\\Aluminium batten"
output_destination =  "C:\\Users\\joshu\\Documents\\Cambridge\\4th Year Work\\Project\\Panel Measurements\\Aluminium batten\\alum_complete_data.csv" 

process_rt60_raw(directory_to_search, output_destination)