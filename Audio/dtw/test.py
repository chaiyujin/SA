from dtw import dtw, print_path
import numpy as np

a = np.asarray([1, 1, 1, 3], dtype=np.double)
b = np.asarray([11, 14], dtype=np.double)
a = np.reshape(a, (a.shape[0], 1))
b = np.reshape(b, (b.shape[0], 1))


cost, path, warp_s, warp_t = dtw(b, a)
print_path(path)
print(warp_t)
print(warp_s)
