import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


#i = 0  1  2  3  4
#
#    0  0  0  0  1.
#   .5  0  0  0  0
#   .5 .5  0  0  0
#    0 .5 .5 .5  0
#    0  0 .5 .5  0

def make_m_with_boundary(n, d=2):
  m = np.zeros([n + 2, n + 2])
  for i in range(n):
    for j in range(d):
      if i + j + 1 > n:
        m[i, i] += 1. / d
      else:
        m[i + j + 1, i] = 1. / d
  m[-1, n] = 1.
  m[-1, n + 1] = 1.
  return m

def make_circular_m(n, d=2):
  m = np.zeros([n + 1, n + 1])
  for i in range(n):
    for j in range(d):
      if i + j + 1 <= n:
        m[i + j + 1, i] = 1. / d
      else:
        m[i, i] += 1. / d
  m[0, -1] = 1.
  return m

def add_warp(m, start, end):
  row = m[..., start, :].copy()
  m[..., start, :] -= row
  m[..., end, :] += row

def add_warps(m, warps):
  for warp in warps:
    add_warp(m, *warp)

def compute_expectation(m):
  e, ev = np.linalg.eig(m)
  nev = ev[..., 0] / np.sum(ev[..., 0], axis=-1)
  expectation = (1. / nev[..., 0]) - 1.

n = 100
d = 6

m_bound = make_m_with_boundary(n, d)
m_circ = make_circular_m(n, d)

warps = [[ 1, 38], [ 4, 14], [ 9, 31], [21, 42], [28, 84], [36, 44], [51, 67],
         [71, 91], [80,100], [16,  6], [47, 26], [49, 11], [56, 53], [62, 19],
         [64, 60], [87, 24], [93, 73], [95, 75], [98, 78]]


perm = np.eye(n, n) * np.ones(n_particles, n, n)
for i in range(n_particles)

# warps.shape ==  [19, 2]

# Want:
# warps.shape = [N, 10, 2]
# m.shape = [N, 101, 101]
# application of warps:
#
# for i in range(10):
#   

add_warps(m_bound, warps)
add_warps(m_circ, warps)


#start = np.zeros(m.shape[0])
#start[0] = 1.
#win = np.zeros(m.shape[0]) 
#win[-1] = 1.
#print("Start:")
#print(start)
#print()
#
#print("Win:")
#print(win)
#print()
#
#state = start.copy()
#for i in range(4):
#  state = m.dot(state)
#  print((i + 1), state)
#
#u, s, v = np.linalg.svd(m)
#with np.printoptions(precision=4, threshold=10000, linewidth=201):
#  print("U")
#  print(u)
#  print()
#
#  print("S")
#  print(s)
#  print()
#
#  print("V")
#  print(v)
#  print()
#
#  e, ev = np.linalg.eig(m)
#  print("Eigenvalues")
#  print(e)
#  print()
#
#  print("Eigenvectors")
#  print(ev)
#  print()
#
#  print(ev.dot(np.diag(e)).dot(ev))
#  print(ev.T.dot(np.diag(e)).dot(ev))
#  print(ev.dot(np.diag(e)).dot(ev.T))
