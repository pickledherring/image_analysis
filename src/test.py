from src.main import *
from ast import literal_eval

data = []
with open("cells2.csv", "r") as f:
    cells_reader = csv.reader(f)
    headers = next(cells_reader)
    for i, row in enumerate(cells_reader):
        data.append([])
        for j, col in enumerate(row):
            try:
                col = literal_eval(col)
            except:
                col = str(col)
            
            data[i].append(col)
            
# print(data[0])
with open("cells.csv", 'a') as f:
    cells_writer = csv.writer(f)

    for row in data:
        cells_writer.writerow(row)