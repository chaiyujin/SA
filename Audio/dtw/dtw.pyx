from __future__ import division
import numpy as np
cimport numpy as np

DTYPE = np.double
ctypedef np.double_t DTYPE_t

ITYPE = np.int
ctypedef np.int_t ITYPE_t


cdef double __dist(np.ndarray[DTYPE_t,ndim=1] x,
                 np.ndarray[DTYPE_t,ndim=1] y):

    cdef int length = len(x)
    cdef unsigned int i
    cdef DTYPE_t d = 0

    for i in range(length):
        d += (x[i] - y[i]) ** 2

    return np.sqrt(d)

def dtw(np.ndarray source, np.ndarray target, distance_fn=None):
    cdef int nrows = source.shape[0]
    cdef int ncols = target.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=2] cost =\
        np.zeros((nrows + 1, ncols + 1), dtype = DTYPE)
    cdef np.ndarray[ITYPE_t, ndim=2] path =\
        np.zeros((nrows + 1, ncols + 1), dtype = ITYPE)

    cost[:, 0] = 1e6
    cost[0, :] = 1e6
    cost[0, 0] = 0.0


    cdef double dist
    cdef double last
    cdef int idx = 0
    for i in range(nrows):
        for j in range(ncols):
            # calc the distance
            if distance_fn is None:
                dist = __dist(source[i], target[j])
            else:
                dist = distance_fn(source[i], target[j])
            # choose the best direction
            last = cost[i, j]
            idx = 3
            for k in range(2, 0, -1):
                ii = i + ((k >> 1) & 1)
                jj = j + (k & 1)
                if cost[ii, jj] < last:
                    last = cost[ii, jj]
                    idx = k
            cost[i + 1, j + 1] = dist + last
            path[i + 1, j + 1] = idx

    # get the path
    cdef int x = nrows
    cdef int y = ncols
    warp_s = []
    warp_t = []
    list_s = []
    list_t = []
    old_x = x
    old_y = y
    while x > 0 or y > 0:
        if path[x, y] == 0:
            break
        else:
            k = path[x, y]
            path[x, y] += 10
            x -= k & 1
            y -= (k >> 1) & 1
            # warp
            if k == 3:
                list_s.append(source[old_x - 1])
                list_t.append(target[old_y - 1])
            elif k == 2:
                list_t.append(target[old_y - 1])
            elif k == 1:
                list_s.append(source[old_x - 1])
            if old_x != x:
                # list_t.reverse()
                warp_t.append(np.array(list_t))
                list_t = []
            if old_y != y:
                # list_s.reverse()
                warp_s.append(np.array(list_s))
                list_s = []
            old_x = x
            old_y = y
    for i in range(nrows + 1):
        for j in range(ncols + 1):
            if path[i, j] < 10:
                path[i, j] = 0
            else:
                path[i, j] -= 10
    warp_s.reverse()
    warp_t.reverse()
    for i in range(len(warp_s)):
        if len(warp_s[i]) == 0:
            warp_s[i] = warp_s[i - 1]

    for i in range(len(warp_t)):
        if len(warp_t[i]) == 0:
            warp_t[i] = warp_t[i - 1]
    return cost, path, warp_s, warp_t

def print_path(path):
    for i in range(path.shape[0]):
        st = ''
        for j in range(path.shape[1]):
            if path[i][j] == 0:
                st += '.'
            elif path[i][j] == 1:
                st += '|'
            elif path[i][j] == 2:
                st += '_'
            else:
                st += '\\'
        print(st)
