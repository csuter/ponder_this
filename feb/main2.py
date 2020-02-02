import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

np.set_printoptions(precision=5, threshold=13000, linewidth=1000)

n = 100
d = 6

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
  return np.real((1. / nev[..., 0]) - 1.)

def init_warps(num_warps):
  warps = []
  for i in range(num_warps):
    used_starts = set([warp[0] for warp in warps])
    available_starts = set(range(n)) - used_starts
    warps.append([np.random.choice(list(available_starts)),
                  np.random.randint(1, n + 1)])
  return warps

def make_proposal(m, warps):
  new_m = make_circular_m(n, d)
  new_warps = warps.copy()
  warp_to_change = np.random.randint(0, len(new_warps))
  used_starts = set([warp[0] for warp in new_warps])
  available_starts = set(range(n)) - used_starts
  new_warps[warp_to_change][0] = np.random.choice(list(available_starts))
  new_warps[warp_to_change][1] = np.random.randint(1, n + 1)
  add_warps(new_m, new_warps)
  return new_m, new_warps

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

m = make_circular_m(n, d)
warps = init_warps(10)
add_warps(m, warps)

n_iters = 50000
target = 66.978705
exps = np.zeros(n_iters)
log_accept_ratios = np.zeros(n_iters)
acceptances = np.zeros(n_iters).astype(np.bool)

prev_exp = compute_expectation(m)

def annealing_schedule(t):
  a = 20
  b = 7
  c = 20
  d = np.ceil(c - c * t)
  return d * np.exp(b * (1. + np.floor(a * t) - a * t))  / np.exp(b)

def _():
  ts = np.linspace(0, 1, 1000)
  plt.plot(ts, annealing_schedule(ts))
  plt.savefig('annealing_schedule.png')
_()

smallest_error = float('inf')
best_warps = None
for i in range(n_iters):
  prop_m, prop_warps = make_proposal(m, warps)
  exp = compute_expectation(prop_m)

  temp = annealing_schedule(float(i) / n_iters)

  log_uniform = np.log(np.random.uniform(0., 1.))
  log_accept_ratios[i] = ((prev_exp - target) ** 2 - (exp - target) ** 2) / temp
  acceptances[i] = (log_uniform < log_accept_ratios[i])
  if acceptances[i]:
    m, warps, prev_exp = prop_m.copy(), prop_warps.copy(), exp
  exps[i] = prev_exp
  err = np.abs(exps[i] - target)
  if err < smallest_error:
    smallest_error = err
    best_warps = warps.copy()
    print(err)
    print(best_warps)

fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize=(20, 10))
axes[0].plot(exps, label='expected lengths')
axes[0].legend()
axes[1].plot(acceptances, label='accept/reject')
axes[1].legend()
axes[2].plot(log_accept_ratios, label='log accept ratios')
axes[2].legend()
plt.savefig('fig.png')

print("Best err:", smallest_error)
print("Best warps:\n", best_warps)
