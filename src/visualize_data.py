import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_traffic(input_file):
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True) 
    
    # Plotting
    plt.figure(figsize=(15, 6))
    
    # Plot first 500 intervals to see details clearly, or all of it
    # Let's plot the first week
    first_week = df.first('7D') # Requires TimeIndex, let's assume it sort of works if sorted
    
    # Check if df has timeindex
    plt.plot(df.index[:400], df['Vehicle_Count'][:400], label='Traffic Density (15 min)')
    
    plt.title('Traffic Density Over Time (First 400 intervals)')
    plt.xlabel('Date/Time')
    plt.ylabel('Vehicle Count')
    plt.legend()
    plt.grid(True)
    
    output_img = 'traffic_density_plot.png'
    plt.savefig(output_img)
    print(f"Plot saved to {output_img}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, '..', 'processed_traffic_data.csv')
    visualize_traffic(input_path)
