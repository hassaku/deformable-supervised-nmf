#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import sklearn.metrics as sk_metrics
from sklearn.base import BaseEstimator, TransformerMixin

class DeformableSupervisedNMF(BaseEstimator, TransformerMixin):
    """
    Reference:
        http://www.slideshare.net/DaichiKitamura/nmf-in-japanese
    """
    def __init__(self,
                 supervised_components_list=None, unknown_componets=None,
                 supervised_max_iter_list=[], unknown_max_iter=100,
                 eta=0.1, mu_list=[0, 0, 0, 0],
                 X_list=[], progress=False):
        """
        :param supervised_components_list:
        :param unknown_componets:
        :param supervised_max_iter_list:
        :param unknown_max_iter:
        :param eta: rate of deforming. default is 0.1.
        :param mu_list: penalty coefficients between following features
                        0: supervised and deformed
                        1: supervised and unknown
                        2: deformed and unknown
                        3: (supervised + deformed) and unknown
        :param X_list:
        :param progress:
        :return:
        """
        if type(supervised_components_list) is not list:
            print "supervised_components_list must be a list."
            sys.exit(-1)

        if len(supervised_max_iter_list) == 0:
            supervised_max_iter_list = [100] * len(supervised_components_list)

        self.__supervised_components_list = supervised_components_list
        self.__unknown_components = unknown_componets
        self.__supervised_max_iter_list = supervised_max_iter_list
        self.__unknown_max_iter = unknown_max_iter

        self.mu_list = mu_list
        self.eta = eta

        self.__eps = 1e-8        # for avoidance of division by zero
        self.__tolerance = 1e-8  # for stopping iteration

        self.progress = progress

        self.supervised_features_list = self.__fit_fg(X_list, max_iter_list=supervised_max_iter_list)
        self.unknown_features = None

    def fit_transform(self, X, y=None):
        return self.__fit_bg_deformable(X)

    def fit(self, X, y=None, **params):
        self.fit_transform(X, y)
        return self

    def __debug(self, msg):
        if self.progress:
            print(msg)

    def __fit_fg(self, X_list, max_iter_list):
        supervised_features_list = []

        for xi, xdata in enumerate(X_list):
            xdata = np.mat(xdata)
            T, D = xdata.shape
            X_abs = abs(xdata)
            H0 = np.mat(np.random.rand(self.__supervised_components_list[xi], D))
            U0 = np.mat(np.random.rand(T, self.__supervised_components_list[xi]))

            for curr_iter in range(max_iter_list[xi]):
                update_U = (H0 * X_abs.T / (self.__eps + (H0 * H0.T) * U0.T)).T
                U = np.multiply(U0, update_U)
                U0 = U

                update_H = (X_abs.T * U / (self.__eps + H0.T * (U.T * U))).T
                H = np.multiply(H0, update_H)
                H0 = H

                rse = np.sqrt(sk_metrics.mean_squared_error(X_abs, U*H))

                max_update = np.max(np.max(update_H))
                if max_update < self.__tolerance:
                    self.__debug("Finished (max_update: %f)" % max_update)
                    break
                self.__debug("%d: %f" % (curr_iter, rse))

            supervised_features_list.append(H0)  # consider with abs(X)

        return supervised_features_list

    def __fit_bg_deformable(self, X):
        Z = np.mat(abs(X)).T
        dims, samples = Z.shape

        fg_basis = np.vstack(self.supervised_features_list)
        fg_basis_dims = fg_basis.shape[0]

        F = np.array(abs(fg_basis)).T
        D = np.array(np.random.rand(dims, fg_basis_dims))
        H = np.array(np.random.rand(dims, self.__unknown_components))
        G = np.array(np.random.rand(fg_basis_dims, samples))
        U = np.array(np.random.rand(self.__unknown_components, samples))

        for it in range(self.__unknown_max_iter):
            self.__debug("D: %f, H: %f, G: %f, U:%f" % (D.max(), H.max(), G.max(), U.max()))
            D, H, G, U, rse, update_value = self.__update(Z, F, D, H, G, U, mu=self.mu_list, eta=self.eta, it=it)

            self.__debug("%d: %f(rse), %f(update_value)" % (it, rse, update_value))
            if update_value < self.__tolerance:
                self.__debug("Finished (last update value: %f)" % update_value)
                break

        bias = 0
        for fgi, fg_dims in enumerate(self.__supervised_components_list):
            self.supervised_features_list[fgi] = fg_basis[bias:fg_dims+bias, :]
            bias += fg_dims

        self.unknown_features = np.dot(np.mat(X).T - np.dot(fg_basis.T, G), U.I).T

        return self.supervised_features_list + [self.unknown_features]

    def __update(self, Z, F, D, H, G, U, mu, eta=0.3, it=None):
        dims, samples = Z.shape

        F = np.mat(F)
        D = np.mat(D)
        H = np.mat(H)
        G = np.mat(G)
        U = np.mat(U)

        V1 = 2 * mu[0] * np.dot(F, np.sum(np.multiply(F, D), axis=0).T)
        V2 = 2 * mu[2] * np.dot(H, np.dot(D.T, H).T)
        V = np.tile(V1, F.shape[1]) + V2

        # update D ###########################
        R = np.dot(F+D, G) + np.dot(H, U)

        ## D plus
        D_numer1 = (eta*F + D)
        D_numer2 = np.dot(np.divide(Z, R + self.__eps), G.T)
        D_numer = np.multiply(D_numer1, D_numer2)

        D_denom1 = np.tile(np.sum(G, axis=1), dims).T
        D_denom2 = 2 * mu[1] * np.dot(H, (np.dot((F + D).T, H)).T)
        D_denom = D_denom1 + V + D_denom2

        D_plus = np.divide(D_numer, D_denom + self.__eps) - eta * F

        ## D minus
        D_numer2 -= V
        D_numer = np.multiply(D_numer1, D_numer2)
        D_denom = D_denom1 + D_denom2

        D_minus = np.divide(D_numer, D_denom + self.__eps) - eta * F

        D0 = np.mat(np.zeros((D.shape[0], D.shape[1])))
        D0[V >= 0] = np.array(D_plus)[V >= 0]
        D0[V < 0] = np.array(D_minus)[V < 0]

        D = D0
        R = np.dot(F+D, G) + np.dot(H, U)

        # update H ###########################
        W = 2 * mu[2] * np.dot(D, np.dot(D.T, H))
        S = 2 * mu[3] * np.dot(F + D, np.dot((F + D).T, H))

        ## H plus
        H_numer1 = H
        H_numer2 = np.dot(np.divide(Z, R + self.__eps), U.T)
        H_numer = np.multiply(H_numer1, H_numer2)

        H_denom1 = np.tile(np.sum(U, axis=1), dims).T
        H_denom2 = 2 * mu[1] * np.dot(F, np.dot(F.T, H))
        H_denom = H_denom1 + H_denom2 + W + S

        H_plus = np.divide(H_numer, H_denom + self.__eps)

        ## H minus
        H_numer2 -= W
        H_numer = np.multiply(H_numer1, H_numer2)
        H_denom = H_denom1 + H_denom2 + S

        H_minus = np.divide(H_numer, H_denom + self.__eps)

        H0 = np.mat(np.zeros((H.shape[0], H.shape[1])))
        H0[W >= 0] = np.array(H_plus)[W >= 0]
        H0[W < 0] = np.array(H_minus)[W < 0]

        update_value = np.nanmax(np.nanmax(np.divide(H0, H + self.__eps)))

        H = H0
        R = np.dot(F+D, G) + np.dot(H, U)

        # update G ###########################
        G_numer1 = np.dot((F + D).T, np.divide(Z, R + self.__eps))
        G_numer = np.multiply(G, G_numer1)

        G_denom = np.tile(np.sum(F + D, axis=0).T, samples)

        G0 = np.divide(G_numer, G_denom + self.__eps)

        # update U ###########################
        U_numer1 = np.dot(H.T, np.divide(Z, R + self.__eps))
        U_numer = np.multiply(U, U_numer1)
        U_denom = np.tile(np.sum(H, axis=0).T, samples)

        U0 = np.divide(U_numer, U_denom + self.__eps)

        # reconstruction err #################
        rse = np.sqrt(sk_metrics.mean_squared_error(Z, (F + D0)*G0 + H0*U0))

        return D0, H0, G0, U0, rse, update_value

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy import signal

    # Generate sample data
    np.random.seed(0)
    n_samples = 200
    time = np.linspace(0, 8, n_samples)

    s1 = np.sin(2 * time)                    # Signal 1: sinusoidal signal
    s2 = np.sign(np.sin(3 * time))           # Signal 2: square signal
    s3 = signal.sawtooth(2 * np.pi * time)   # Signal 3: saw tooth signal

    def __format_data(x):
        y = np.tile(x, [5, 1])
        y += 0.2 * np.random.normal(size=y.shape)  # Add noise
        y /= y.std(axis=0)
        y -= y.min()
        return y

    tx1 = __format_data(s1)
    tx2 = __format_data(s2)
    ux = __format_data(s3)

    dsnmf = DeformableSupervisedNMF(supervised_components_list=[5, 5], unknown_componets=5,
                                    supervised_max_iter_list=[1000]*2, unknown_max_iter=5000,
                                    eta=0.01, mu_list=[0.01, 0.01, 0.01, 0.01],
                                    X_list=[tx1, tx2], progress=True)

    dx1 = 1.2 * __format_data(s1)
    dx2 = 0.8 * __format_data(s2)

    fx1, fx2, fx3 = dsnmf.fit_transform(dx1 + dx2 + ux)

    plt.rcParams.update({'axes.titlesize': 'small'})
    plt.subplot(434)
    plt.plot(tx1.T)
    plt.tick_params(labelbottom='off')
    plt.title('pretrained signals')
    plt.subplot(4, 3, 7)
    plt.plot(tx2.T)
    plt.tick_params(labelbottom='off')

    plt.subplot(4, 3, 3)
    plt.plot((dx1 + dx2 + ux).T)
    plt.tick_params(labelbottom='off')
    plt.title('input signals')

    plt.subplot(4, 3, 5)
    plt.plot(dx1.T)
    plt.tick_params(labelbottom='off')
    plt.title('truth signals')
    plt.subplot(4, 3, 8)
    plt.plot(dx2.T)
    plt.tick_params(labelbottom='off')
    plt.subplot(4, 3, 11)
    plt.plot(ux.T)
    plt.title('(unknown)')
    plt.tick_params(labelbottom='off')

    plt.subplot(4, 3, 6)
    plt.plot(fx1.T)
    plt.tick_params(labelbottom='off')
    plt.title('decomposing signals')
    plt.subplot(4, 3, 9)
    plt.plot(fx2.T)
    plt.tick_params(labelbottom='off')
    plt.subplot(4, 3, 12)
    plt.plot(fx3.T)
    plt.tick_params(labelbottom='off')

    plt.show()
