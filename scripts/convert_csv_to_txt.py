import math
import pandas as pd

csv_path = r"E:\Shamman Files\AI projects\bangla GPT\datasets\Bengali_Banglish_80K_Dataset.csv"

data = pd.read_csv(csv_path)

# Export to text file, each row in a new line
with open(r'E:\Shamman Files\AI projects\Andrej Karpathy\building gpt from scratch\datasets\banglish.txt', 'w+', 
          encoding="utf-8") as file:
    for index, row in data.iterrows():
        if isinstance(row['Banglish'], str):
            file.write(row['Banglish'])
            file.write('\n \n')