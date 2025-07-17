import numpy as np

def peaks_to_fingerprints(pks, F):
  peaks = sorted(pks, key=lambda x : (x[1], x[0]))
  dists = []
  
  '''
  [(41, 3), (2047, 6), (41, 10), (2044, 14), (41, 18), (2043, 22), (41, 25), (2048, 29), (41, 33), (2013, 37), (2046, 
  37), (41, 40)]
  '''

  t_early = 0

  for i in range(len(peaks)):
    proxim = []
    j = i + 1
    t_early = i
    while j < len(peaks) and len(proxim) < F:
      proxim.append(j)
      if peaks[j][1] != peaks[t_early][1]:
        t_early = j
      j += 1
    
    while len(proxim) > F:
      if abs(proxim[-1][0] - proxim[i][0]) < abs(proxim[t_early][0] - proxim[i][0]):
        proxim.pop(t_early)
      else:
        proxim.pop(-1)
    
    dists.extend([((peaks[i][0], peaks[proxim[j]][0], peaks[proxim[j]][1] - peaks[i][1]), peaks[i][1]) for j in range(len(proxim))])

  return dists
