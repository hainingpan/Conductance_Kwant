import numpy as np
import tinyarray


s0 = tinyarray.array([[1, 0], [0, 1]]);
sx = tinyarray.array([[0, 1], [1, 0]]);
sy = tinyarray.array([[0, -1j], [1j, 0]]);
sz = tinyarray.array([[1, 0], [0, -1]]);

t0 = tinyarray.array([[1, 0], [0, 1]]);
tx = tinyarray.array([[0, 1], [1, 0]]);
ty = tinyarray.array([[0, -1j], [1j, 0]]);
tz = tinyarray.array([[1, 0], [0, -1]]);


t0s0 = np.kron(s0,s0);
t0sx = np.kron(s0,sx);
t0sy = np.kron(s0,sy);
t0sz = np.kron(s0,sz);

txs0 = np.kron(sx,s0);
txsx = np.kron(sx,sx);
txsy = np.kron(sx,sy);
txsz = np.kron(sx,sz);

tys0 = np.kron(sy,s0);
tysx = np.kron(sy,sx);
tysy = np.kron(sy,sy);
tysz = np.kron(sy,sz);

tzs0 = np.kron(sz,s0);
tzsx = np.kron(sz,sx);
tzsy = np.kron(sz,sy);
tzsz = np.kron(sz,sz);



