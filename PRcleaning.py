import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

def parse_measurement_code(code):
    """
    Extracts the base name and the repeat number from a measurement code.
    Example: '2CcForward 2' -> ('2CcForward', '2')
    Example: '1A2NHifi' -> ('1A2NHifi', '1')
    """
    code_str = str(code).strip()
    # Looks for any text, followed by optional spaces, ending with digits
    match = re.search(r'^(.*?)\s*(\d+)$', code_str)
    if match:
        return match.group(1).strip(), match.group(2)
    else:
        return code_str, '1'

def review_and_average_rt60(input_csv, output_csv):
    # 1. Load the collated data
    df = pd.read_csv(input_csv)
    
    # Parse Base Measurement (Combination) and Repeat number from Measurement Code
    parsed = df['Measurement Code'].apply(parse_measurement_code)
    df['Combination'] = [p[0] for p in parsed]
    df['Repeat'] = [p[1] for p in parsed]
    
    # 2. Separate frequency columns from metadata
    meta_cols = ['Filename', 'Measurement Code', 'Measurement Type', 'Combination', 'Repeat']
    freq_cols = [c for c in df.columns if c not in meta_cols and c != 'full']
    
    # Convert frequency column names to floats for the x-axis mapping
    freq_numeric = [float(f) for f in freq_cols]
    
    combinations = df['Combination'].unique()
    indices_to_drop = []
    
    print("\n--- INTERACTIVE RT60 REVIEW ---")
    print("For each grouped measurement, a plot will appear.")
    print("-> To drop a specific metric, use 'Metric-Repeat' (e.g., 'T30-2' or 'EDT-4').")
    print("-> To drop an ENTIRE repeat, just enter the number (e.g., '2').")
    print("-> You can mix them separated by commas: 'T30-1, 2, EDT-3'")
    print("-> Press Enter without typing anything to keep all data.")
    print("-------------------------------\n")
    
    for combo in combinations:
        subset = df[df['Combination'] == combo]
        repeats = subset['Repeat'].unique()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics_to_plot = ['EDT (s)', 'T20 (s)', 'T30 (s)']
        colors = {'EDT (s)': 'blue', 'T20 (s)': 'green', 'T30 (s)': 'red'}
        line_styles = ['-', '--', ':', '-.']
        
        has_data = False
        for i, rep in enumerate(repeats):
            rep_subset = subset[subset['Repeat'] == rep]
            ls = line_styles[i % len(line_styles)]
            
            for metric in metrics_to_plot:
                metric_data = rep_subset[rep_subset['Measurement Type'] == metric]
                if not metric_data.empty:
                    has_data = True
                    # Get the values for the frequency columns
                    y_vals = metric_data[freq_cols].values[0]
                    # Convert to float (handles any accidental strings)
                    y_vals = [float(y) if pd.notnull(y) else 0.0 for y in y_vals]
                    
                    ax.plot(freq_numeric, y_vals, 
                            label=f"{metric[:3]} (Rep {rep})", # Shorthand label (e.g. EDT)
                            color=colors.get(metric, 'black'), 
                            linestyle=ls, marker='o', markersize=4)
        
        if not has_data:
            plt.close(fig)
            continue
            
        ax.set_title(f"Grouped Measurement: {combo}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Time (s)")
        ax.set_xscale('log')
        ax.set_xticks(freq_numeric)
        ax.set_xticklabels(freq_cols, rotation=45)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Show plot without blocking script execution
        plt.show(block=False)
        plt.pause(0.1) # Forces the UI to render the plot before asking for input
        
        # Prompt user
        user_input = input(f"Reviewing '{combo}'. Enter drops (or press Enter to keep all): ").strip()
        
        if user_input:
            drop_commands = [cmd.strip() for cmd in user_input.split(',')]
            for cmd in drop_commands:
                if '-' in cmd:
                    # Dropping specific metric for a repeat (e.g., 'T30-2')
                    parts = cmd.split('-')
                    if len(parts) == 2:
                        met_part, rep_part = parts[0].strip().upper(), parts[1].strip()
                        
                        # Map user shorthand to actual metric names in CSV
                        met_map = {'EDT': 'EDT (s)', 'T20': 'T20 (s)', 'T30': 'T30 (s)'}
                        actual_metric = met_map.get(met_part, met_part)
                        
                        to_drop = subset[(subset['Repeat'] == rep_part) & 
                                         (subset['Measurement Type'] == actual_metric)].index
                        indices_to_drop.extend(to_drop)
                        print(f"  -> Flagged {actual_metric} for Repeat {rep_part} for removal.")
                else:
                    # Dropping entire repeat
                    rep_part = cmd.strip()
                    to_drop = subset[subset['Repeat'] == rep_part].index
                    indices_to_drop.extend(to_drop)
                    print(f"  -> Flagged ALL metrics for Repeat {rep_part} for removal.")
        else:
            print("  -> Keeping all data for this combination.")
            
        plt.close(fig)

    # 5. Drop the flagged measurements
    cleaned_df = df.drop(index=indices_to_drop)
    
    # 6. Calculate Averages
    print("\nCalculating averages...")
    
    # Columns to average (frequencies + 'full' band)
    cols_to_avg = freq_cols.copy()
    if 'full' in df.columns:
        cols_to_avg.append('full')
        
    # Convert all freq columns to numeric just to be absolutely safe before math
    for col in cols_to_avg:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        
    # Group by Base Measurement (Combination) and Measurement Type, then calculate mean
    # We also group by Filename so we don't lose track of where it came from
    avg_df = cleaned_df.groupby(['Filename', 'Combination', 'Measurement Type'])[cols_to_avg].mean().reset_index()
    
    # Rename Combination back to a more sensible header
    avg_df.rename(columns={'Combination': 'Base Measurement'}, inplace=True)
    
    # 7. Export the final CSV
    avg_df.to_csv(output_csv, index=False)
    print(f"Success! Cleaned and averaged data exported to:\n{output_csv}")

# --- EXECUTION ---
# Using the exact path setup from your previous request
input_path = r"C:\Users\joshu\Documents\Cambridge\4th Year Work\Project\Practice Room\collated_data.csv"
output_path = r"C:\Users\joshu\Documents\Cambridge\4th Year Work\Project\Practice Room\cleaned_averaged_data.csv"

review_and_average_rt60(input_path, output_path)
