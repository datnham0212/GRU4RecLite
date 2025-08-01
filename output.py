# import torch

# # Load the .pt file
# file_path = "yoochoose-data\save_model.pt"
# file_contents = torch.load(file_path)

# # Print the contents to inspect it
# print(file_contents)
import pandas as pd

# Load the three files
file1 = pd.read_csv("yoochoose-data/yoochoose-buys.tsv", sep="\t")
file2 = pd.read_csv("yoochoose-data/yoochoose-test.tsv", sep="\t")
file3 = pd.read_csv("yoochoose-data/yoochoose-clicks.tsv", sep="\t")

# Find the common session IDs
common_sessions = set(file1['SessionId']) & set(file2['SessionId']) & set(file3['SessionId'])

print("Common Session IDs:", common_sessions)