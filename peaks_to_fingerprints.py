import numpy as np

def peaks_to_fingerprints(spec, F):

  '''
  Runs in O(n^4), optimized version would be O(n^2 logn)

  Input: Spec
      2D array of shape (N, M)
          N: # of frequency bins
          M: # of time bins
  Input: F
      the fanout value

  Output: peaks
      array of tuples [((fi, fj, dt),t), ...]
  '''

  grid = np.array(spec)
  peaks = []

  for t in range(grid.shape[1]):
    for f in range(grid.shape[0]):
      if (grid[f,t] == 0):
        continue
      neighbors = []
      col = t

      while len(neighbors) < F and col < grid.shape[1]:
        arr = []
        for i in range(grid.shape[0]):
          if i != f and grid[i, col] == 1:
            arr.append((col, i))
        
        while len(neighbors) + len(arr) > F:
          if abs(arr[-1][0] - f) < abs(arr[0][0] - f):
            arr.pop(0)
          else:
            arr.pop(-1)
        neighbors.extend(arr)
        col += 1
      peaks.extend([((f, neighbors[i][0], neighbors[i][1] - t), t) for i in range(len(neighbors))])

  return peaks
