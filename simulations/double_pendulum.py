import numpy as np, scipy.integrate as spi, matplotlib.pyplot as plt, time
import pygame

G = 9.8
L = 1

def pendulumHamiltonian(t, cvs):
  """Takes in the values for the canonical variables for the pendulum in 2D (i.e. [theta, omega]) and returns the time derivates (i.e. [dtheta/dt, domega/dt]"""
  return [cvs[1], -(G/L)*np.sin(cvs[0])]

if __name__ == "__main__":
  cvs = [np.pi/2, 0] #theta = 1 rad, omega = 0 rad/s
  solution = spi.RK45(pendulumHamiltonian, t0=0, y0=cvs, t_bound=100, max_step=1/100)

  xs = []
  ys = []
  for i in range(100):
    solution.step()
    x, y = np.sin(solution.y[0]), 1-np.cos(solution.y[0])
    xs.append(x)
    ys.append(y)

  plt.plot(xs, ys, 'o')
  plt.show()

