import numpy as np


A = np.loadtxt("/Users/sofiastepanova/Desktop/лабы_парал-е/matrix_A.txt", skiprows=1)
B = np.loadtxt("/Users/sofiastepanova/Desktop/лабы_парал-е/matrix_B.txt", skiprows=1)
C = np.loadtxt("/Users/sofiastepanova/Desktop/лабы_парал-е/matrix_C", skiprows=1)

C_true = A @ B

print("C++ result:")
print(C)

print("\nPython result:")
print(C_true)

if np.allclose(C, C_true):
    print("\nOK: results are correct ")
else:
    print("\nERROR: results differ ")