import numpy as np
def obj_cluster(res,T,rho):
    T_transpose=T.conj().transpose()

    # a * b =>a.dot(b)matrix multiply
    # a. * b => a * b  element - wise multiply
    G = T_transpose.dot(T)
    eps = 10 ** -8
    rowLen = len(G)
    colLen = len(G)
    Gtilda=np.zeros((rowLen, colLen))
    for i in range(rowLen):
        for j in range(colLen):
            aaa=(G[i, j] / (G[i, i] + eps) + (G[i, j] / (G[j, j]) + eps))
            Gtilda[i, j] = 0.5 * (G[i, j] / (G[i, i] + eps) + (G[i, j] / (G[j, j]) + eps))

        Gtilda[i, i] = G[i, i] / colLen;

    G = Gtilda;

    I = np.eye(rowLen);
    # tao = max(0, -min(np.linalg.eig(G))) + 10
    tao=10

    f1=np.zeros((rowLen,rowLen))
    for i in range(rowLen):
        for j in range(rowLen):
            f1[i, j] = res[i] * res[j] * (G[j, i] + tao * I[j, i])

    f2=np.zeros(rowLen)
    for i in range(rowLen):
        f2[i] = res[i] * res[i]


    f2 = sum(f2);

    f = -sum(sum(f1)) + tao * f2;

    hh = 0;

    for i in range(rowLen):
        ccc=res[i]
        hh = hh + np.log(1 + abs(res[i]) / eps)

    f = f + (rho / (np.log(1 + 1 / eps))) * hh

    return f



