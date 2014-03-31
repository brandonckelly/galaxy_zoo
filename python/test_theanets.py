#!/usr/bin/env python

__author__ = 'brandonkelly'

import numpy as np
import os
import sys
import time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

import climate
import theanets

climate.enable_default_logging()

ngrid = 100
xgrid, ygrid = np.mgrid[:ngrid, :ngrid]
f1 = np.sin(xgrid / 10.0) * np.cos(ygrid / 10.0)
f2 = np.sqrt(np.abs(np.cos(xgrid / 50.0) * np.sin(ygrid / 50.0)))

features = np.column_stack((xgrid.astype(float).ravel(), ygrid.astype(float).ravel()))
features -= features.mean(axis=0)
# features /= features.std(axis=0)
response = np.column_stack((f1.ravel(), f2.ravel()))

train_set_x, valid_set_x, train_set_y, valid_set_y = train_test_split(features, response)

n_hidden = 100
layers = (features.shape[1], n_hidden, n_hidden, n_hidden, response.shape[1])
experiment = theanets.Experiment(theanets.Regressor, layers=layers, train_batches=100)

# experiment.add_dataset('train', (train_set_x, train_set_y))
# experiment.add_dataset('valid', (valid_set_x, valid_set_y))

experiment.run(train=(train_set_x, train_set_y), valid=(valid_set_x, valid_set_y))

prediction = experiment.network.predict(features)

print 'Training error:', np.mean(np.sum((prediction - response) ** 2, axis=1))

fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
surf1 = ax1.plot_surface(xgrid, ygrid, f1, cmap='hot', cstride=2, rstride=2, shade=True)

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
surf2 = ax2.plot_surface(xgrid, ygrid, prediction[:, 0].reshape((ngrid, ngrid)), cmap='hot', cstride=2,
                         rstride=2, shade=True)

plt.show()

fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
surf1 = ax1.plot_surface(xgrid, ygrid, f2, cmap='hot', cstride=2, rstride=2, shade=True)

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
surf2 = ax2.plot_surface(xgrid, ygrid, prediction[:, 1].reshape((ngrid, ngrid)), cmap='hot', cstride=2,
                         rstride=2, shade=True)

plt.show()
