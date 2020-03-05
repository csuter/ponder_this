import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys

np.set_printoptions(precision=5, threshold=13000, linewidth=1000)

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
  return np.real((1. / nev[..., 0]) - 1.)

warps = eval(sys.argv[1])

m = make_circular_m(100, 6)
add_warps(m, warps)

print(compute_expectation(m))
