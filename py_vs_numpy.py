import numpy as np
import time

import numpy as np
import time

def add_matrix(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A))] for i in range(len(A))]

def subtract_matrix(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A))] for i in range(len(A))]

def strassen_matrix_multiplication(A, B):
    n = len(A)
    
    # Base case when size of matrix is 1x1
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    
    # Splitting the matrices into quadrants
    mid = n // 2
    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]
    
    B11 = [row[:mid] for row in B[:mid]]
    B12 = [row[mid:] for row in B[:mid]]
    B21 = [row[:mid] for row in B[mid:]]
    B22 = [row[mid:] for row in B[mid:]]
    
    # Strassen's algorithm
    M1 = strassen_matrix_multiplication(add_matrix(A11, A22), add_matrix(B11, B22))
    M2 = strassen_matrix_multiplication(add_matrix(A21, A22), B11)
    M3 = strassen_matrix_multiplication(A11, subtract_matrix(B12, B22))
    M4 = strassen_matrix_multiplication(A22, subtract_matrix(B21, B11))
    M5 = strassen_matrix_multiplication(add_matrix(A11, A12), B22)
    M6 = strassen_matrix_multiplication(subtract_matrix(A21, A11), add_matrix(B11, B12))
    M7 = strassen_matrix_multiplication(subtract_matrix(A12, A22), add_matrix(B21, B22))
    
    # Combining the results
    C11 = add_matrix(subtract_matrix(add_matrix(M1, M4), M5), M7)
    C12 = add_matrix(M3, M5)
    C21 = add_matrix(M2, M4)
    C22 = add_matrix(subtract_matrix(add_matrix(M1, M3), M2), M6)
    
    # Combining quadrants into a single matrix
    C = []
    for i in range(mid):
        C.append(C11[i] + C12[i])
    for i in range(mid):
        C.append(C21[i] + C22[i])
    
    return C

def pad_matrix(A, size):
    padded = [[0]*size for _ in range(size)]
    for i in range(len(A)):
        for j in range(len(A[i])):
            padded[i][j] = A[i][j]
    return padded

# Generate two 1000x1000 matrices with random numbers
A = np.random.rand(1000, 1000).tolist()
B = np.random.rand(1000, 1000).tolist()

# Find the nearest power of two greater than or equal to 1000
size = 1
while size < 1000:
    size *= 2

# Function to multiply two matrices using plain Python
def plain_python_matrix_multiplication(A, B):
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

# Generate two 1000x1000 matrices with random numbers
A = np.random.rand(1000, 1000).tolist()
B = np.random.rand(1000, 1000).tolist()

# Pad the matrices to the nearest power of two
A_padded = pad_matrix(A, size)
B_padded = pad_matrix(B, size)

# Measure time for Strassen's matrix multiplication
start_time = time.time()
strassen_result_padded = strassen_matrix_multiplication(A_padded, B_padded)
# Extract the relevant part of the result
strassen_result = [row[:1000] for row in strassen_result_padded[:1000]]
strassen_duration = time.time() - start_time

print(f"Strassen's matrix multiplication took {strassen_duration:.2f} seconds.")

# Measure time for plain Python matrix multiplication
start_time = time.time()
plain_result = plain_python_matrix_multiplication(A, B)
plain_duration = time.time() - start_time

print(f"Plain Python matrix multiplication took {plain_duration:.2f} seconds.")

# Convert lists to NumPy arrays
A_np = np.array(A)
B_np = np.array(B)

# Measure time for NumPy matrix multiplication
start_time = time.time()
numpy_result = np.dot(A_np, B_np)
numpy_duration = time.time() - start_time

print(f"NumPy matrix multiplication took {numpy_duration:.2f} seconds.")
