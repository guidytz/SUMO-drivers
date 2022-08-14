import numpy as np

arr = np.array([[0.0, 6, 5]])

re = 1

med = np.mean(arr, axis=0)
print(f"{type(med)}")
print(f"{med}")

if type(med) == np.float64:
    print("no")
else:
    print("ye")

# [20.6, 3.4]