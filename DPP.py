import numpy as np
from scipy.linalg import eigh

class DualDPP:
    def __init__(self, C, B):
        self.C = C
        self.B = B
        self.D_all, self.V_all = eigh(C)

    def elem_sympoly(self, k):
        ''' Algorithm 7 '''
        N = len(self.D)
        E = np.zeros((k+1,N+1))
        E[0,:] = 1
        for l in range(1, k+1):
            for n in range(1, N+1):
                E[l, n] = E[l, n-1] + self.D[n-1]*E[l-1,n-1];
        return E

    def k_dpp_phase1(self, k):
        ''' Algorithm 8 '''
        E = self.elem_sympoly(k)
        i = len(self.D)-1
        rem = k-1
        S = np.zeros(k)
        while rem >= 0:
            # compute marginal of i given that we choose remaining values from 1:i
            marginal = 1 if i==rem else self.D[i]*E[rem,i]/E[rem+1,i+1]
            
            # sample marginal
            if np.random.random() < marginal:
                S[rem] = i
                rem -= 1
            i -= 1
        return list(map(int, np.sort(S)))

    def makeV(self, cols):
        normalise = lambda v: v/np.dot(np.dot(self.C, v), v)
        V = np.zeros((self.V_all.shape[0], sum(cols)))
        for i, [col] in enumerate(np.argwhere(cols)):
            V[:, i] = normalise(self.V_all[:, col])
        self.V = V

    def phase1(self):
        self.D = self.D_all/( 1 + self.D_all )
        J = np.random.rand(len(self.D)) < self.D
        self.makeV(J)

    def sample_dual(self, k=None):
        N = self.B.shape[1]
        self.D, self.V = self.D_all, self.V_all
        if k == None:
            self.phase1()
            k = self.V.shape[1]
        else:
            J = self.k_dpp_phase1(k)
            cols = np.array([True if i in J else False for i in range(len(self.D))])
            self.makeV(cols)

        # PHASE 2
        Y = np.zeros(k, dtype=int)
        for c in range(k-1,-1,-1):
            # choose vector index, with prob proportional to K_ii = v_i^T v_i, lambda==1??
            P = np.sum(np.power(np.dot(self.V.T, self.B), 2), axis=0)
            r = np.random.choice(range(N), p=P/np.sum(P))
            Y[c] = r

            # Select an eigenvector to remove
            vi = np.nonzero(np.dot(self.V.T, self.B[:, r]))[0][0]
            v = np.copy(self.V[:,vi])
            self.V = np.delete(self.V, vi, axis=1)
            # Update K to condition on event r in Y
            self.V -= np.outer(np.dot(self.V.T, self.B[:, r]), v).T/np.dot(v, self.B[:, r])

            # Orthogonalize wrt <v1, v2> = v1T C v2
            if c > 0:
                for a in range(c):
                    for b in range(a):
                        self.V[:, a] -= np.dot(np.dot(self.C, self.V[:, a]), self.V[:, b])*self.V[:, b]
                    self.V[:, a] /= np.linalg.norm(self.V[:, a])
        return np.sort(Y)