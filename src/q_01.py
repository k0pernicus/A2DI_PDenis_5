import numpy as np

def compute_q0(matrix):

    m_l = len(matrix)
    q_0 = np.zeros(m_l)
    for state in range(m_l):
        q_0[state] = (1/m_l)

    return q_0

def power_method(matrix):

    q_old = np.zeros(len(matrix))
    q_0 = compute_q0(matrix)

    iterations = 0
    while not np.allclose(q_old, q_0):
        q_old = q_0
        tmp = np.dot(q_0.T, matrix)
        q_0 = np.divide(tmp, np.linalg.norm(tmp))
        iterations += 1

    return q_0, iterations

def compute_stochastic_matrix(matrix):
    """

    """

    for i, line in enumerate(matrix):
        s = sum(line)
        for j, element in enumerate(line):
            matrix[i][j] /= s

    return matrix

def main():
    
    A = np.array([[4.,5.],[6.,5.]])
    B = np.array([[-4.,10.],[7.,5.]])

    qA_0, iterations_A = power_method(A)
    qB_0, iterations_B = power_method(B)

    print("qA: {} - {}".format(qA_0, iterations_A))
    print("qB: {} - {}".format(qB_0, iterations_B))

if __name__ == '__main__':
    main()