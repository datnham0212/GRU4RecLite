import pandas as pd

# Input and output file paths
input_file = 'yoochoose-data/yoochoose-clicks.tsv'
output_file = 'yoochoose-data/yoochoose-clicks-reduced.tsv'

# Load the TSV file
data = pd.read_csv(input_file, sep='\t')

# Specify the 3 session IDs to keep
session_ids_to_keep = [670038, 5362338, 6502006]  # Replace with desired session IDs

# Filter the data to only include rows with the specified session IDs
reduced_data = data[data['SessionId'].isin(session_ids_to_keep)]

# Save the reduced data to a new TSV file
reduced_data.to_csv(output_file, sep='\t', index=False)

print(f"Reduced {input_file} to only include 3 session IDs and saved to {output_file}")