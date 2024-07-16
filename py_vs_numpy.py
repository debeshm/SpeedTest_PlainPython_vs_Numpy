import numpy as np
import time

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
