import numpy as np


s0 = np.array([[1.0, 0.0], [0.0, 1.0]]);
sx = np.array([[0.0, 1.0], [1.0, 0.0]]);
sy = np.array([[0.0, -1j], [1j, 0.0]]);
sz = np.array([[1.0, 0.0], [0.0, -1.0]]);

t0 = np.array([[1.0, 0.0], [0.0, 1.0]]);
tx = np.array([[0.0, 1.0], [1.0, 0.0]]);
ty = np.array([[0.0, -1j], [1j, 0.0]]);
tz = np.array([[1.0, 0.0], [0.0, -1.0]]);


s0t0 = np.kron(s0,t0);
s0tx = np.kron(s0,tx);
s0ty = np.kron(s0,ty);
s0tz = np.kron(s0,tz);

sxt0 = np.kron(sx,t0);
sxtx = np.kron(sx,tx);
sxty = np.kron(sx,ty);
sxtz = np.kron(sx,tz);

syt0 = np.kron(sy,t0);
sytx = np.kron(sy,tx);
syty = np.kron(sy,ty);
sytz = np.kron(sy,tz);

szt0 = np.kron(sz,t0);
sztx = np.kron(sz,tx);
szty = np.kron(sz,ty);
sztz = np.kron(sz,tz);

