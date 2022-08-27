import os

a = "aa/yee.txt"

b = os.path.splitext(a.split("/")[-1])[0]

print(b)