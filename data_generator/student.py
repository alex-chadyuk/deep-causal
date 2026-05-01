# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
import os
import pandas as pd

# Set the path to the file you'd like to load
file_path = kagglehub.dataset_download("sharmajicoder/students-academic-performance")
files = os.listdir(file_path)
print(files)

# Read the csv file
df = pd.read_csv(os.path.join(file_path, files[0]))
print("First 5 records", df.head())
