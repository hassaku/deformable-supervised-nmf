#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import

from unittest import TestCase
from nose.tools import ok_, eq_
from deformable_supervised_nmf import DeformableSupervisedNMF
import numpy as np

class DeformableSupervisedNmfTestCase(TestCase):
    def setUp(self):
        self.x = np.sin(np.linspace(0, 8, 100))

    def tearDown(self):
        pass

    def test_init(self):
        ok_(
            DeformableSupervisedNMF(supervised_components_list=[5,10], unknown_componets=5,
                                    supervised_max_iter_list=[10]*2, unknown_max_iter=10,
                                    eta=0.01, mu_list=[0.01, 0.01, 0.01, 0.01], X_list=[self.x, self.x])
        )

    def test_fit(self):
        sdim = 5
        udim = 10
        dsnmf = DeformableSupervisedNMF(supervised_components_list=[sdim], unknown_componets=udim,
                                        supervised_max_iter_list=[10], unknown_max_iter=10,
                                        eta=0.01, mu_list=[0.01, 0.01, 0.01, 0.01], X_list=[self.x])
        eq_(dsnmf.fit_transform(self.x)[0].shape[0], sdim)
        eq_(dsnmf.fit_transform(self.x)[1].shape[0], udim)
