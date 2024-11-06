# create a mini version of the input excel file for quickly testing changes to the code

import pandas as pd
import random

input_file = r"C:\Users\dell3\source\repos\school-data-scraper-4\Freelancer_Data_Mining_Project.xlsx"
output_file = r"C:\Users\dell3\source\repos\school-data-scraper-4\Freelancer_Data_Mining_Project_mini.xlsx"

# Read the Excel file into a dictionary of DataFrames
excel_data = pd.read_excel(input_file, sheet_name=None)

# Process each sheet
for sheet_name, df in excel_data.items():
    if sheet_name in ["Baseball Conferences", "Softball Conferences", "DNU_States"]:
        excel_data[sheet_name] = df
        continue
    # If the sheet has less than 20 rows, keep all rows
    if len(df) < 7:
        excel_data[sheet_name] = df
    else:
        # Randomly select 20 rows
        random_rows = random.sample(range(len(df)), 7)
        excel_data[sheet_name] = df.iloc[random_rows]

# Write the modified data to a new Excel file
writer = pd.ExcelWriter(output_file)
for sheet_name, df in excel_data.items():
    df.to_excel(writer, sheet_name=sheet_name, index=False)
writer._save()

print(f"Mini Excel file created: {output_file}")