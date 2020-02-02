import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt

def make_m(n, d=2):
  m = np.zeros([n + 2, n + 2])
  for i in range(n):
    for j in range(d):
      if i + j + 1 > n:
        m[i, i] += 1. / d
      else:
        m[ i + j + 1, i] = 1. / d
  m[-1, n] = 1.
  m[-1, n + 1] = 1.
  return m

def add_warp(m, start, end):
  col = m[:, start].copy()
  m[:, start] -= col
  m[:, end] += col

def add_warps(m, warps):
  for warp in warps:
    add_warp(m, *warp)

def plot_m(m):
  plt.imshow(m, interpolation='none')
  plt.colorbar()
  nx = m.shape[0]
  x_positions = np.arange(0, nx) # pixel count at label position
  labels = ['S'] + list(range(1, nx-1)) + ['E']
  labels = ['S'] + list(range(1, nx-1)) + ['E']
  plt.xticks(x_positions, labels)
  plt.yticks(x_positions, labels)
  plt.ylabel("Beginning position")
  plt.xlabel("Landing position")
  plt.show()

def compute_exp(m, n=1000):
  state = np.zeros(m.shape[0])
  state[0] = 1.

  exps = np.zeros(n)
  e = 0
  for i in range(n):
    state = m.dot(state)
    e += (i + 1) * state[-2]
    exps[i] = e
  return exps

def compute_exp_closed_form(m):
  state = np.zeros(m.shape[0])
  state[0] = 1.
  proj = np.zeros(m.shape[0]) 
  proj[-2] = 1.

  a = np.eye(m.shape[0]) - m
  return proj.dot(np.linalg.solve(a, state))

def plot_exps(exps):
  with np.printoptions(precision=6):
    strexp = str(exps[-1:])
  plt.figure(figsize=(10, 6))
  plt.plot(exps, label='Rolling Expectation', alpha=.7)
  plt.axhline(exps[-1], label='Final expectation = {}'.format(strexp), c='r', alpha=.7)
  plt.legend()
  plt.show()


# Example
def run_example():
  m = make_m(100, 6)
  warps = [[ 1, 38], [ 4, 14], [ 9, 31], [21, 42], [28, 84], [36, 44], [51, 67],
           [71, 91], [80,100], [16,  6], [47, 26], [49, 11], [56, 53], [62, 19],
           [64, 60], [87, 24], [93, 73], [95, 75], [98, 78]]
  add_warps(m, warps)
  exps = compute_exp(m)
  cf_exp = compute_exp_closed_form(m)
  print(exps[-1])
  print(cf_exp)
  #plot_exps(exps)
#run_example()


m = make_m(4, 2)
print("M:")
print(m)
print()

start = np.zeros(m.shape[0])
start[0] = 1.
win = np.zeros(m.shape[0]) 
win[-2] = 1.
print("Start:")
print(start)
print()

print("Win:")
print(win)
print()

state = start.copy()
for i in range(4):
  state = m.dot(state)
  print((i + 1), state)

u, s, v = np.linalg.svd(m)
with np.printoptions(precision=4, threshold=10000, linewidth=201):
  print("U")
  print(u)
  print()

  print("S")
  print(s)
  print()

  print("V")
  print(v)
  print()

  e, ev = np.linalg.eig(m)
  print("Eigenvalues")
  print(e)
  print()

  print("Eigenvectors")
  print(ev)
  print()

  print(ev.dot(np.diag(e)).dot(np.linalg.inv(ev)))
  #print(ev.T.dot(np.diag(e)).dot(ev))
  #print(ev.dot(np.diag(e)).dot(ev.T))
