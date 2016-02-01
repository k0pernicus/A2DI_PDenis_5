#-*- coding: utf-8 -*-

import numpy as np

def compute_q0(matrix):

    m_l = len(matrix)
    q_0 = np.zeros(m_l)
    for state in range(m_l):
        q_0[state] = (1/m_l)

    return q_0

def power_method(p_bare_bare):

    q_old = np.zeros(len(p_bare_bare))
    q_0 = compute_q0(p_bare_bare)

    iterations = 0
    while not np.allclose(q_old, q_0):
        q_old = q_0
        tmp = np.dot(q_0.T, p_bare_bare)
        q_0 = np.divide(tmp, np.linalg.norm(tmp))
        iterations += 1

    return q_0, iterations

def compute_stochastic_matrix(matrix):

    for i, line in enumerate(matrix):
        s = sum(line)
        for j, element in enumerate(line):
            matrix[i][j] /= s

    return matrix

def compute_p_bare(p):

    p_l = len(p)

    # Calcul de a
    a = np.zeros(p_l)
    for i, line in enumerate(p):
        if sum(line) == 0:
            a[i] = 1

    # Calcul de u
    u = np.array(p_l)
    u.fill(1./p_l)

    return np.add(p, (np.dot(a, u.T)))

def compute_p_bare_bare(p_bare, alpha = 0.85):

    p_l = len(p_bare)

    # Calcul de u
    u = np.ones(p_l)
    u.fill(1./p_l)

    e = np.ones(p_l)

    return np.add(np.dot(alpha, p_bare), np.dot((1 - alpha), np.dot(e, u.T)))

def main():
    W = np.array([[0.,1.,1.],[0.,0.,1.],[0.,1.,0.]])
    P = compute_stochastic_matrix(W)

    print(P)

    p_bare = compute_p_bare(P)
    p_bare_bare = compute_p_bare_bare(p_bare)

    print(power_method(p_bare_bare))

if __name__ == '__main__':
    main()
