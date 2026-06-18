import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

d = np.load('metric_depth/results_raw/second_billboard1-After_raw_depth_meter.npy')
print('shape', d.shape, 'min', d.min(), 'max', d.max(), 'median', np.median(d))
print('zeros', (d==0).sum(), '(', 100.0*(d==400).sum()/d.size, '% )')

h,w = d.shape
ys,xs = np.where(d>0)
pts = np.vstack([xs, ys]).T
tree = cKDTree(pts)
u,v = 948,295
dist, idx = tree.query([u,v])
xu,yu = pts[idx]
print('nearest valid coord', xu, yu, 'depth', d[yu, xu])


import pandas as pd
df = pd.read_csv('scale.csv')
for i, row in df.iterrows():
    u, v, gt = row.iloc[0], row.iloc[1], row.iloc[2]
    # ensure numeric types for the KD-tree query and integer indices for image access
    dist, idx = tree.query([float(u), float(v)])
    xu, yu = pts[idx]
    est_depth = d[int(yu), int(xu)]
    print(f'Point {i}: GT={gt}, Estimated={est_depth}, Distance to nearest valid pixel={dist}')

plt.imshow(d==400, cmap='Reds', alpha=0.5)
plt.scatter([u], [v], color='red', s=100, marker='x')
plt.scatter([xu], [yu], color='blue', s=100, marker='x')
plt.title('invalid mask')
plt.show()