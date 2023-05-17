import random
import numpy as np


def reduce_svd(U, S, V):
    # Convert U, D, V to NumPy arrays if they are lists
    U = np.array(U)
    S = np.array(S)
    V = np.array(V)

    # Set the tolerance level for singular values close to zero
    tolerance = 1e-10

    # Find the indices of singular values that are significant
    indices = np.where(S > tolerance)[0]

    # Reduce the matrices to the significant singular values and vectors
    U_reduced = U[:, indices]
    D_reduced = S[indices]
    V_reduced = V[indices, :]

    return U_reduced, D_reduced, V_reduced


def power_iter_svd(input_matrix):
    def dot(A, B):
        if type(A[0]) != list:
            A = [A]
        if type(B[0]) != list:
            B = [[b] for b in B]
        ret = [[0] * len(B[0]) for i in range(len(A))]

        for row in range(len(ret)):
            for col in range(len(ret[0])):
                ret[row][col] = 0

                for i in range(len(B)):
                    ret[row][col] += A[row][i] * B[i][col]

        if len(ret) == 1 and len(ret[0]) == 1:
            return ret[0][0]
        elif len(ret[0]) == 1:
            return [r[0] for r in ret]
        return ret

    def transpose(A):
        if type(A[0]) != list:
            A = [A]
        rows = len(A)
        cols = len(A[0])
        B = [[0] * rows for i in range(cols)]

        for row in range(rows):
            for col in range(cols):
                B[col][row] = A[row][col]
        return B


    # Squared input matrix (A*A^T)
    input_squared = dot(input_matrix, transpose(input_matrix))

    # Number of iterations
    iterations = 100

    # Number of SVDs to recover
    N = min(len(input_squared), len(input_squared[0]))

    # Return values
    # Left signular vectors
    U = [[0] * len(input_squared[0]) for i in range(N)]

    # Singular values
    S = [0] * N

    for n in range(N):
        # Randomly initialize search vector
        b = [random.random() for i in range(len(input_squared[0]))]

        dominant_svd = None
        for k in range(iterations):
            # Input matrix multiplied by b_k
            projection = dot(input_squared, b)

            # Norm of input matrix multiplied by b_k
            norm = dot(projection, projection) ** 0.5

            # Calculate b_{k+1}
            b_next = [d / norm for d in projection]
            dominant_svd = dot(b, projection) / dot(b, b)

            b = b_next

        S[n] = dominant_svd ** 0.5

        for i in range(len(b)):
            U[i][n] = b[i]

        outer_product = [[0] * len(b) for j in range(len(b))]
        for i in range(len(b)):
            for j in range(len(b)):
                outer_product[i][j] = dominant_svd * b[i] * b[j]

        for i in range(len(input_squared)):
            for j in range(len(input_squared[0])):
                input_squared[i][j] -= outer_product[i][j]

    Dinv = [[0] * N for i in range(N)]
    for i in range(N):
        Dinv[i][i] = 1 / S[i]

    V_T = dot(Dinv, dot(transpose(U), input_matrix))
    return U, S, V_T


if __name__ == '__main__':
    input_matrix = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]
    U, S, V_T = power_iter_svd(input_matrix)
    U, S, V_T = reduce_svd(U, S, V_T)

