import pandas as pd
import re
import os

def process_rew_files(file_paths, output_csv):
    all_rows = []
    
    # Define the exact metrics we want to keep. 
    # 'r' columns and the text-based phase column are excluded.
    metrics_to_keep = [
        'EDT (s)', 'T20 (s)', 'T30 (s)', 'Topt (s)', 
        'ToptStart (dB)', 'ToptEnd (dB)', 'T60M (s)', 
        'C50 (dB)', 'C80 (dB)', 'D50 (%)', 'TS (s)'
    ]
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split the text by the REW header to isolate individual measurements
        blocks = content.split('RT60 data saved by REW')
        
        for block in blocks:
            if not block.strip():
                continue
                
            # Extract Measurement Name
            meas_match = re.search(r'Measurement:\s*(.*)', block)
            if not meas_match:
                continue
            raw_meas = meas_match.group(1).strip()
            
            # Strip date/time if present (e.g., matching " 06/11/25 10:51")
            meas_code = re.sub(r'\s+\d{2}/\d{2}/\d{2}\s+\d{2}:\d{2}$', '', raw_meas).strip()
            
            # Dictionary to hold the extracted data for this block: {metric: {freq: value}}
            block_data = {metric: {} for metric in metrics_to_keep}
            
            # Parse the lines for the data
            lines = block.strip().split('\n')
            parsing_data = False
            
            for line in lines:
                line = line.strip()
                if line.startswith('Format is'):
                    parsing_data = True
                    continue
                
                if parsing_data and line:
                    # Filter for 1/3 octave bands (includes numeric frequencies and the 'full' band)
                    if ' 1/3 ' in line:
                        # Tie "Zero Phase" together so it doesn't shift the metrics coming after it
                        safe_line = line.replace('Zero Phase', 'Zero_Phase')
                        parts = safe_line.split()
                        
                        freq = parts[0]
                        
                        # Map the parts, skipping the 'r' indices and the phase text (index 13)
                        if len(parts) >= 18:
                            block_data['EDT (s)'][freq] = parts[2]
                            block_data['T20 (s)'][freq] = parts[4]
                            block_data['T30 (s)'][freq] = parts[6]
                            block_data['Topt (s)'][freq] = parts[8]
                            block_data['ToptStart (dB)'][freq] = parts[10]
                            block_data['ToptEnd (dB)'][freq] = parts[11]
                            block_data['T60M (s)'][freq] = parts[12]
                            block_data['C50 (dB)'][freq] = parts[14]
                            block_data['C80 (dB)'][freq] = parts[15]
                            block_data['D50 (%)'][freq] = parts[16]
                            block_data['TS (s)'][freq] = parts[17]
            
            # Convert the parsed dictionary into structured rows
            for metric in metrics_to_keep:
                # Only append if we actually found data for this metric
                if block_data[metric]:
                    row = {
                        'Filename': filename,
                        'Measurement Code': meas_code,
                        'Measurement Type': metric
                    }
                    row.update(block_data[metric])
                    all_rows.append(row)
                
    # Create the DataFrame
    df = pd.DataFrame(all_rows)
    
    # Sort the columns so metadata is first, followed by numeric frequencies, with 'full' at the end
    meta_cols = ['Filename', 'Measurement Code', 'Measurement Type']
    freq_cols = [c for c in df.columns if c not in meta_cols]
    
    def sort_key(x):
        try:
            return float(x)
        except ValueError:
            return float('inf') # Pushes 'full' to the far right side
            
    sorted_freq_cols = sorted(freq_cols, key=sort_key)
    df = df[meta_cols + sorted_freq_cols]
    
    # Export to CSV
    df.to_csv(output_csv, index=False)
    print(f"Extraction complete. Data saved to {output_csv}")

# --- Execution ---
files_to_process = [
    r"C:\Users\joshu\Documents\Cambridge\4th Year Work\Project\Practice Room\Practice Room measurements\Grouped data.txt", 
    r"C:\Users\joshu\Documents\Cambridge\4th Year Work\Project\Practice Room\Practice Room 2nd Set of Measurements\2Crotation collection.txt", 
    r"C:\Users\joshu\Documents\Cambridge\4th Year Work\Project\Practice Room\Practice Room 2nd Set of Measurements\Standardsetcollected data.txt"
]

output_path = r"C:\Users\joshu\Documents\Cambridge\4th Year Work\Project\Practice Room\collated_data.csv"

process_rew_files(files_to_process, output_path)
