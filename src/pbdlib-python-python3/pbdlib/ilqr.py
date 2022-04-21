import numpy as np
from .poglqr import LQR
import pbdlib as pbd


class ILQR(object):
    def __init__(self, A=None, B=None, nb_dim=2, dt=0.01, horizon=50):
        self._horizon = horizon
        self.A = A
        self.B = B
        self.dt = dt

        self.nb_dim = nb_dim

        self._s_xi, self._s_u = None, None
        self._x0 = None

        self._gmm_xi, self._gmm_u = None, None
        self._mvn_sol_xi, self._mvn_sol_u = None, None

        self._seq_xi, self._seq_u = None, None

        self._S, self._v, self._K, self._Kv, self._ds, self._cs, self._Qc = \
            None, None, None, None, None, None, None

        self._Q, self._z = None, None

        self._x_nom = None
        self._u_nom = None

    @property
    def x_nom(self):
        return self._x_nom

    @x_nom.setter
    def x_nom(self, value):
        self._x_nom = value

    @property
    def u_nom(self):
        return self._u_nom

    @u_nom.setter
    def u_nom(self, value):
        self._u_nom = value

    @property
    def K(self):
        assert self._K is not None, "Solve Ricatti before"

        return self._K

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, value):
        """
        value :
            (ndim_xi, ndim_xi) or
            ((N, ndim_xi, ndim_xi), (nb_timestep, )) or
            (nb_timestep, ndim_xi, ndim_xi)
        """
        self._Q = value

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value):
        """
        value :
            (ndim_xi, ) or
            ((N, ndim_xi, ), (nb_timestep, )) or
            (nb_timestep, ndim_xi)
        """
        self._z = value

    @property
    def Qc(self):
        assert self._Qc is not None, "Solve Ricatti before"

        return self._Qc

    @property
    def cs(self):
        """
        Return d list where control command u is
            u = -K x + d

        :return:
        """
        if self._cs is None:
            self._cs = self.get_feedforward()

        return self._cs

    @property
    def horizon(self):
        return self._horizon

    @horizon.setter
    def horizon(self, value):
        self._horizon = value

    @property
    def u_dim(self):
        """
        Number of dimension of input
        :return:
        """
        if self.B is not None:
            return self.B.shape[1]
        else:
            return self.nb_dim

    @property
    def x_dim(self):
        """
        Number of dimension of state
        :return:
        """
        if self.A is not None:
            return self.A.shape[0]
        else:
            return self.nb_dim * 2

    @property
    def gmm_xi(self):
        """
        Distribution of state
        :return:
        """
        return self._gmm_xi

    @gmm_xi.setter
    def gmm_xi(self, value):
        """
        :param value 		[pbd.GMM] or [(pbd.GMM, list)]
        """
        # resetting solution
        self._mvn_sol_xi = None
        self._mvn_sol_u = None
        self._seq_u = None
        self._seq_xi = None

        self._gmm_xi = value

    @property
    def gmm_u(self):
        """
        Distribution of control input
        :return:
        """
        return self._gmm_u

    @gmm_u.setter
    def gmm_u(self, value):
        """
        :param value 		[float] or [pbd.MVN] or [pbd.GMM] or [(pbd.GMM, list)]
        """
        # resetting solution
        self._mvn_sol_xi = None
        self._mvn_sol_u = None
        self._seq_u = None
        self._seq_xi = None

        if isinstance(value, float):
            self._gmm_u = pbd.MVN(
                mu=np.zeros(self.u_dim), lmbda=10 ** value * np.eye(self.u_dim))
        else:
            self._gmm_u = value

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, value):
        # resetting solution
        self._mvn_sol_xi = None
        self._mvn_sol_u = None

        self._x0 = value

    def get_Q_z(self, t):
        """
        get Q and target z for time t
        :param t:
        :return:
        """
        if self._gmm_xi is None:
            z, Q = None, None

            if self._z is None:
                z = np.zeros(self.A.shape[-1])
            elif isinstance(self._z, tuple):
                z = self._z[0][self._z[1][t]]
            elif isinstance(self._z, np.ndarray):
                if self._z.ndim == 1:
                    z = self._z
                elif self._z.ndim == 2:
                    z = self._z[t]

            if isinstance(self._Q, tuple):
                Q = self._Q[0][self._Q[1][t]]
            elif isinstance(self._Q, np.ndarray):

                if self._Q.ndim == 2:
                    Q = self._Q
                elif self._Q.ndim == 3:
                    Q = self._Q[t]

            return Q, z
        else:
            if isinstance(self._gmm_xi, tuple):
                gmm, seq = self._gmm_xi
                return gmm.lmbda[seq[t]], gmm.mu[seq[t]]
            elif isinstance(self._gmm_xi, pbd.GMM):
                return self._gmm_xi.lmbda[t], self._gmm_xi.mu[t]
            elif isinstance(self._gmm_xi, pbd.MVN):
                return self._gmm_xi.lmbda, self._gmm_xi.mu
            else:
                raise ValueError("Not supported gmm_xi")

    def get_R(self, t):
        if isinstance(self._gmm_u, pbd.MVN):
            return self._gmm_u.lmbda
        elif isinstance(self._gmm_u, tuple):
            gmm, seq = self._gmm_u
            return gmm.lmbda[seq[t]]
        elif isinstance(self._gmm_u, pbd.GMM):
            return self._gmm_u.lmbda[t]
        else:
            raise ValueError("Not supported gmm_u")

    def get_A(self, t):
        if self.A.ndim == 2:
            return self.A
        else:
            return self.A[t]

    def get_B(self, t):
        if self.B.ndim == 2:
            return self.B
        else:
            return self.B[t]

    def backward_pass(self):
        """
        https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf
        :return:
        """

        #
        _S = [None for i in range(self._horizon)]
        _v = [None for i in range(self._horizon)]
        _K = [None for i in range(self._horizon - 1)]
        _Kv = [None for i in range(self._horizon - 1)]
        _Qc = [None for i in range(self._horizon - 1)]

        _P[-1] = self.get()
        _p[-1] = Q.dot(z)

        for t in range(self.horizon - 2, -1, -1):
            Q, z = self.get_Q_z(t)
            R = self.get_R(t)
            A = self.get_A(t)
            B = self.get_B(t)

            _Qc[t] = np.linalg.inv(R + B.T.dot(_S[t + 1]).dot(B))
            _Kv[t] = _Qc[t].dot(B.T)
            _K[t] = _Kv[t].dot(_S[t + 1]).dot(A)

            AmBK = A - B.dot(_K[t])

            _S[t] = A.T.dot(_S[t + 1]).dot(AmBK) + Q
            _v[t] = AmBK.T.dot(_v[t + 1]) + Q.dot(z)

        self._S = _S
        self._v = _v
        self._K = _K
        self._Kv = _Kv
        self._Qc = _Qc

        self._ds = None
        self._cs = None

    def get_target(self):
        ds = []

        for t in range(0, self.horizon - 1):
            ds += [np.linalg.inv(self._S[t].dot(self.A)).dot(self._v[t])]

        return np.array(ds)

    def get_feedforward(self):
        cs = []

        for t in range(0, self.horizon - 1):
            cs += [self._Kv[t].dot(self._v[t + 1])]

        return np.array(cs)

    def get_first_command(self, xi0):
        return self.get_command(xi0, 0)

    def get_command(self, xi0, i):
        return -self._K[i].dot(xi0) + self._Kv[i].dot(self._v[i])

    def get_seq(self, xi0, return_target=False):
        xis = [xi0]
        us = [-self._K[0].dot(xi0) + self._Kv[0].dot(self._v[0])]

        ds = []

        for t in range(1, self.horizon-1):
            A = self.get_A(t)
            B = self.get_B(t)
            xis += [A.dot(xis[-1]) + B.dot(us[-1])]

            if return_target:
                d = np.linalg.inv(self._S[t].dot(A)).dot(self._v[t])
                ds += [d]

                us += [self._K[t].dot(d - xis[-1])]
            else:
                us += [-self._K[t].dot(xis[-1]) + self._Kv[t].dot(self._v[t + 1])]

        if return_target:
            return np.array(xis), np.array(us), np.array(ds)
        else:
            return np.array(xis), np.array(us)
