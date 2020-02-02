import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

np.set_printoptions(precision=5, threshold=13000, linewidth=1000)

n = 100
d = 6

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


exps = np.zeros((n, n))

#for start in range(1, n):
#  print(start)
#  for end in range(1, n):
#    m = make_circular_m(n, d)
#    add_warp(m, start, end)
#    exps[start, end] = compute_expectation(m)
#np.save('hdiw/data.npy', exps)

exps = np.load('hdiw/data.npy')

print(exps[1, 1:])

plt.figure(figsize=(20, 20))
plt.imshow(exps[1:, 1:], interpolation='none')
plt.colorbar()
plt.savefig('hdiw/all.png')
plt.close()

print(((exps[:, 1] - exps[:, 99]) / 100)[:, None])

#for start in [50]:
#  print(start)
#  for end in range(1, n):
#    m = make_circular_m(n, d)
#    add_warp(m, start, end)
#    print(compute_expectation(m))
#  print()
#  print()
