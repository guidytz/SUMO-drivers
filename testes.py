import os

a = "sumo_graph/data/ola.csv"

b = a.split("/")
for i in b:
    if ".csv" in i:
        c = os.path.splitext(i)[0]

print(c)
