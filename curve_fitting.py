import os
print(__file__)
print(os.getcwd())
print(os.path.dirname(os.path.realpath(__file__)))

os.chdir(os.path.dirname(os.path.realpath(__file__)))
print(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import sciann as sn

#training data
x_data, y_data = np.meshgrid(
    np.linspace(-np.pi, np.pi, 50),
    np.linspace(-np.pi, np.pi, 50)
)
f_data = np.sin(x_data) * np.sin(y_data)
fig_1 = plt.figure(figsize=(6,5))
im = plt.pcolormesh(x_data, y_data, f_data, cmap='seismic')
plt.colorbar(im)
plt.plot()

# #step 1 : Setting up the neural network
# x = sn.Variable('x')
# y = sn.Variable('y')
# f = sn.Functional('f', [x, y], [10, 10, 10, 10], 'tanh') # field variable, input variables, hidden_layer, act. func
# f_pred = f.eval([x_data, y_data])
# fig_2 = plt.figure(figsize=(6,5))
# im = plt.pcolormesh(x_data, y_data, f_pred, cmap='seismic')
# plt.colorbar(im)
# plt.plot()
#
# #step 2 : Setting up the optimization model
# d1 = sn.Data(f)
# m = sn.SciModel([x, y], d1)
# m.summary()
#
# h = m.train([x_data, y_data], f_data, learning_rate=0.002, epochs=200, verbose=1)
# fig_3 = plt.figure(figsize=(6,5))
# plt.semilogy(h.history['loss'])
# plt.plot()
#
# x_test, y_test = np.meshgrid(np.linspace(-2*np.pi, 2*np.pi, 100), np.linspace(-2*np.pi, 2*np.pi, 100))
# f_test = np.sin(x_test) * np.sin(y_test)
#
# f_pred = f.eval([x_test, y_test])
# fig_4, ax = plt.subplots(1, 2)
#
# im = ax[0].pcolor(x_test, y_test, f_test, cmap='seismic')
# plt.colorbar(im, ax=ax[0])
# im = ax[1].pcolor(x_test, y_test, f_pred, cmap='seismic')
# plt.colorbar(im, ax=ax[1])
# plt.plot()

"""
==============================
Physics-informed deep learning
==============================
"""
#step 1 : Defining the network
x = sn.Variable('x')
y = sn.Variable('y')
f = sn.Functional('f', [x, y], [10, 10, 10, 10], 'tanh') #fields, variables, hidden_layer, act. func

#step 2 : Defining the objectives fucntions and optimization model
f_xx = sn.math.diff(f, x, order=2)
f_yy = sn.math.diff(f, y, order=2)
L = f_xx + f_yy + 2*f

d1 = sn.Data(f)
d2 = sn.Data(L)
m = sn.SciModel([x, y], [d1, d2], loss_func='mse')
h = m.train([x_data, y_data],
            [f_data, 'zero'],
            epochs=500,
            batch_size=50,
            learning_rate=0.002,
            verbose=1)

m.save_weights('trained-curve_fitting.hdf5')
m.load_weights('trained-curve_fitting.hdf5')

fig_5 = plt.figure(figsize=(6,5))
plt.semilogy(h.history['loss'])
plt.plot()

x_test, y_test = np.meshgrid(np.linspace(-2*np.pi, 2*np.pi, 100), np.linspace(-2*np.pi, 2*np.pi, 100))
f_test = np.sin(x_test) * np.sin(y_test)

f_pred = f.eval(m, [x_test, y_test])
f_pred2 = m.predict([x_test, y_test])


fig_6, ax = plt.subplots(1, 4, figsize=(10, 5))
im = ax[0].pcolor(x_test, y_test, f_test, cmap='seismic')
plt.colorbar(im, ax=ax[0])
im = ax[1].pcolor(x_test, y_test, f_pred, cmap='seismic')
plt.colorbar(im, ax=ax[1])
im = ax[2].pcolor(x_test, y_test, f_pred2[0], cmap='seismic')
plt.colorbar(im, ax=ax[2])
im = ax[3].pcolor(x_test, y_test, f_pred2[1], cmap='seismic')
plt.colorbar(im, ax=ax[3])
plt.plot()

plt.show()
