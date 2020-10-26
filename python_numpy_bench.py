# 00_time_test.py
# https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/two_layer_net_numpy.ipynb#scrollTo=05mkjrUTj_vx
# https://github.com/cicerolneto/tutorials/tree/master/beginner_source/examples_tensor

# C:\Users\cicer_ymv1igm\AppData\Local\Microsoft\WindowsApps\python3.8.exe

from datetime import datetime
start_time = datetime.now()
# INSERT YOUR CODE
# https://www.it-swarm.dev/pt/python/medir-o-tempo-decorrido-em-python/940039357/


# -*- coding: utf-8 -*-


import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6

for t in range(10000):
#or t in range(500):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2



print('')
time_elapsed = datetime.now() - start_time

print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

print('')
import sys
print(sys.version)
print("Versão atual do Python: %s" % sys.version.split(" ")[0])
print('')

print ('Win10 GD900-1(   GD900-1 ) 0:00:15.135759', 'Python 3.8.3, PowerShell  # x10.000')  ## HP i7
print ('Windows 10  :(hh:mm:ss.ms) 0:00:24.669147', 'Python 3.8,   PowerShell  # x10.000')  ## HP i7
print('')
print ('Debian 10.4 GD900-1                      ', 'Miniconda, Python 3.8.2  # x10.000')  ## HP i7
print ('Debian 10.4 :(hh:mm:ss.ms) 0:00:25.781554', 'Miniconda, Python 3.8.5  # x10.000')  ## HP i7

print ('Ubuntu 20.04:(hh:mm:ss.ms) 0:00:18.850015', 'Miniconda, Python 3.8.3  # x10.000')  ## HP i7
print ('MX-19.2_ahs :(hh:mm:ss.ms) 0:00:20.965619', 'Miniconda, Python 3.8.3  # x10.000')  ## HP i7







print('')
print('')
print ('Debian 10.4 :(hh:mm:ss.ms) 0:00:43.824153 - 0.0 - 0.0 amperes', 'Debian Evolute i5 m480 - TMUX, all monitores, Miniconda, Python 3.8.3  # x10.000')
print ('Debian 11 bullseye GD900-1)0:00:36.999054 - 0.0 - 0.0 amperes', 'Debian Evolute i5 m480 - Bash , Miniconda, Python 3.8.3  # x10.000')

print ('Debian 10.4 :(hh:mm:ss.ms) 0:00:43.963636 - 0.0 - 0.0 amperes', 'Debian Inspiron N5010 i5 m460 - TMUX, all monitores, Miniconda, Python 3.8.3  # x10.000')
print('')

print ('Debian 10.4 :(hh:mm:ss.ms) 0:01:27.193811 - 2.4 - 2.7 amperes (sem tela)', 'CCE T5750 - TMUX, all monitores, Miniconda, Python 3.8.3  # x10.000')

print ('Debian 10.4 :(hh:mm:ss.ms) 0:00:53.231128 - 2.6 - 2.9 amperes (sem tela)', 'Positivo T7500 - TMUX, all monitores, Miniconda, Python 3.8.3  # x10.000')
print ('Debian 11 bullseye GD900-1)0:00:51.905066 - 0 -   0   amperes (sem tela)', 'Positivo T7500 - Bash,  Miniconda, Python 3.8.5  # x10.000')


print('')

print('')
print ('Debian 10.4 :(hh:mm:ss.ms) 0:00:01.401472', 'Miniconda, Python 3.8.2  # x500 ')
print ('Windows 10  :(hh:mm:ss.ms) 0:00:01.484289', 'Python 3.8 # x500')




