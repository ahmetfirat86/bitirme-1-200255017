import pandas as pd
import numpy as np
import os

def preprocess_data(input_file, output_file, interval='15min'):
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
        return

    # Convert DateTime column to pandas datetime objects
    print("Converting dates...")
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    # Set DateTime as index
    df.set_index('DateTime', inplace=True)

    # We want to count vehicles per interval to get 'Density'
    # We can group by Junction_ID and resample
    # For this dataset, we'll assume we are training for Junction_ID = 1 (since most data seems to be there or we filter)
    # Let's check if there are multiple junctions.
    junctions = df['Junction_ID'].unique()
    print(f"Found Junctions: {junctions}")

    all_resampled = []

    for j_id in junctions:
        print(f"Processing Junction {j_id}...")
        temp_df = df[df['Junction_ID'] == j_id]
        
        # Resample logic: Count frequency of records in the interval
        # resample().size() or .count()
        resampled = temp_df.resample(interval).size().rename('Vehicle_Count')
        
        # Create a dataframe from series
        resampled_df = resampled.to_frame()
        resampled_df['Junction_ID'] = j_id
        
        all_resampled.append(resampled_df)

    # Combine back potentially
    final_df = pd.concat(all_resampled)
    
    # Fill missing values with 0 (no traffic)
    final_df['Vehicle_Count'] = final_df['Vehicle_Count'].fillna(0)
    
    # Save processed data
    print(f"Saving processed data to {output_file}...")
    final_df.to_csv(output_file)
    print("Done!")
    
    # Show first few rows
    print(final_df.head())

if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming this script is in /src and processed data goes to parent or same dir
    # Input is in parent dir 'traffic_data_guncel.csv'
    input_path = os.path.join(base_dir, '..', 'traffic_data_guncel.csv')
    output_path = os.path.join(base_dir, '..', 'processed_traffic_data.csv')
    
    preprocess_data(input_path, output_path)
