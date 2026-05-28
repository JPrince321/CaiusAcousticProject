import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def review_and_average_rt60(input_csv, output_csv):
    """
    Loads raw RT60 data, interactively plots EDT, T20, and T30 for user review, 
    removes specified outliers, and exports the averaged data.
    """
    if not os.path.exists(input_csv):
        print(f"Error: Could not find the input file at {input_csv}")
        return

    df = pd.read_csv(input_csv)
    
    # 1. Separate frequency columns from metadata
    non_freq_cols = ['File', 'Condition', 'Location', 'Repeat', 'Metric', 'full']
    freq_cols = [c for c in df.columns if c not in non_freq_cols]
    freq_numeric = [float(f) for f in freq_cols]
    
    # 2. Create a unified display name for the interactive loop's plots
    df['Display_Name'] = df['Location'] + " - " + df['Condition']
    unique_measurements = df['Display_Name'].unique()
    indices_to_drop = []
    
    print("\n--- INTERACTIVE RT60 REVIEW ---")
    print("For each measurement, a plot will appear.")
    print("-> To drop a specific metric, use 'Metric-Repeat' (e.g., 'T30-2' or 'EDT-4').")
    print("-> To drop an ENTIRE repeat, just enter the number (e.g., '2').")
    print("-> You can mix them separated by commas: 'T30-1, 2, EDT-3'")
    print("If all look good, just press Enter.")
    
    # 3. Loop through each unique Location/Condition pair
    for measurement in unique_measurements:
        subset = df[df['Display_Name'] == measurement]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.canvas.manager.set_window_title(f"Reviewing: {measurement}")
        
        metrics_to_plot = ['EDT', 'T20', 'T30']
        
        for ax, metric in zip(axes, metrics_to_plot):
            metric_data = subset[subset['Metric'] == metric]
            
            for _, row in metric_data.iterrows():
                rep = str(row['Repeat'])
                y_vals = row[freq_cols].astype(float).values
                ax.plot(freq_numeric, y_vals, label=f'Repeat {rep}', marker='o')
            
            ax.set_title(f"{measurement} - {metric}")
            ax.set_xscale('log')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Time (s)')
            
            ax.set_xticks([63, 125, 250, 500, 1000, 2000, 4000, 8000])
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            
            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.legend()
            
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1) 
        
        # 4. Get user input
        user_in = input(f"\nReviewing [{measurement}]. Enter data to drop (or press Enter to keep all): ")
        
        if user_in.strip():
            bad_items = [item.strip() for item in user_in.split(',')]
            
            for item in bad_items:
                if '-' in item:
                    try:
                        metric_part, rep_part = item.split('-', 1)
                        metric_part = metric_part.strip()
                        rep_part = rep_part.strip()
                        
                        to_drop = subset[(subset['Repeat'].astype(str) == rep_part) & 
                                         (subset['Metric'] == metric_part)].index
                        indices_to_drop.extend(to_drop)
                        print(f"  -> Flagged {metric_part} for Repeat {rep_part} for removal.")
                    except ValueError:
                        print(f"  -> WARNING: Could not parse '{item}'. Skipping.")
                else:
                    rep_part = item.strip()
                    to_drop = subset[subset['Repeat'].astype(str) == rep_part].index
                    indices_to_drop.extend(to_drop)
                    print(f"  -> Flagged ALL metrics for Repeat {rep_part} for removal.")
        else:
            print("  -> Keeping all data for this measurement.")
            
        plt.close(fig)

    # 5. Drop the bad measurements and the temporary Display_Name column
    cleaned_df = df.drop(index=indices_to_drop).drop(columns=['Display_Name'])
    
    # 6. Calculate Averages grouping by the new columns
    print("\nCalculating averages...")
    
    cols_to_avg = freq_cols.copy()
    if 'full' in df.columns:
        cols_to_avg.append('full')
        
    avg_df = cleaned_df.groupby(['Condition', 'Location', 'Metric'])[cols_to_avg].mean().reset_index()
    
    # 7. Export the final CSV
    avg_df.to_csv(output_csv, index=False)
    print(f"Success! Cleaned and averaged data exported to:\n{output_csv}")

# --- EXECUTION ---
# Ensure these paths match where your extraction script is saving the data
input_path =  "C:\\Users\\joshu\\Documents\\Cambridge\\4th Year Work\\Project\\Panel Measurements\\Aluminium batten\\alum_complete_data.csv" 
output_path = "C:\\Users\\joshu\\Documents\\Cambridge\\4th Year Work\\Project\\Panel Measurements\\Aluminium batten\\alum_complete_cleaned_data.csv" 
review_and_average_rt60(input_path, output_path)