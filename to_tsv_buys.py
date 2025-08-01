import pandas as pd

# Input and output file paths
input_file = 'yoochoose-data/yoochoose-buys.dat'
output_file = 'yoochoose-data/yoochoose-buys.tsv'

# Read the .dat file (assuming it's comma-separated)
data = pd.read_csv(input_file, delimiter=',', header=None, low_memory=False)

# Assign column names
data.columns = ['SessionId', 'Timestamp', 'ItemId', 'Price', 'Quantity']

# Convert Timestamp to proper datetime format
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Sort by SessionId and Timestamp to maintain order
data.sort_values(by=['SessionId', 'Timestamp'], inplace=True)

# Save as .tsv
data.to_csv(output_file, sep='\t', index=False)

print(f"Converted {input_file} to {output_file}")
