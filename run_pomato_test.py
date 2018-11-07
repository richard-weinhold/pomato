from market_tool import MarketTool
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA


def contingency_Ab_old(contingency):
    A = []
    b = []
    for ptdf in contingency:
        ram_array = np.array([[cap, -cap] for cap in lines.maxflow.values])
        ram_pos = ram_array[:, 0]
        ram_neg = ram_array[:, 1]
        tmp_A = np.concatenate((ptdf, -ptdf), axis=0)
        tmp_b = np.concatenate((ram_pos, -ram_neg), axis=0)
        A += tmp_A.tolist()
        b += tmp_b.tolist()
    A = np.array(A, dtype=np.float16)
    b = np.array(b, dtype=np.float16)
    return A, b

def contingency_Ab(contingency):
    A = np.array([], dtype=np.float16)
    b = np.array([])
    for ptdf in contingency:
        ram_array = np.array([[cap, -cap] for cap in lines.maxflow.values])
        ram_pos = ram_array[:, 0]
        ram_neg = ram_array[:, 1]
        tmp_A = np.array(np.concatenate((ptdf, -ptdf), axis=0), dtype=np.float16)
        tmp_b = np.array(np.concatenate((ram_pos, -ram_neg), axis=0), dtype=np.float16)
        if A.size==0:
            A = tmp_A
            b = tmp_b
        else:
            A = np.concatenate((A, tmp_A), axis=0)
            b = np.concatenate((b, tmp_b), axis=0)
    return A, b

def contingency_Ab_new(contingency):
    A = np.vstack((np.vstack([ptdf, -ptdf]) for ptdf in contingency))
    ram_array = np.array([[cap, -cap] for cap in lines.maxflow.values])
    b = np.hstack(np.concatenate([ram_array[:, 0], -ram_array[:, 1]], axis=0) for i in range(0,len(contingency)))
    return np.array(A, dtype=np.float16), np.array(b, dtype=np.float16)


mato = MarketTool(opt_file="opt_setup.json", model_horizon=range(200,201))
#mato.load_matpower_case('case300', autobuild_gridmodel=True)

mato.load_data_from_file('data/dataset_de.xlsx', autobuild_gridmodel=True)
lines = mato.data.lines

#contingency = mato.grid.n_1_ptdf
print("Ab")
A, b = contingency_Ab_new(mato.grid.n_1_ptdf)
b = np.array(b).reshape(len(b), 1)

print("ranges")
ranges = []
step_size = int(2.5e5)
for i in range(0, int(len(b)/step_size)):
    ranges.append(range(i*step_size, (i+1)*step_size))
ranges.append(range((i+1)*step_size, len(b)))
len(ranges)
print("getting vertices")
vertices = np.array([], dtype=np.int32)
for r in ranges:
    print("jip")
    A_dash = A[r, :]
    b_dash = b[r]
    D = A_dash/b_dash
    model = PCA(n_components=6).fit(D)
    D_t = model.transform(D)
    k = ConvexHull(D_t, qhull_options="QJ")
    vertices = np.append(vertices, k.vertices + r[0])


np.append(np.array([], dtype=np.int32), k.vertices)

#
#len(k.vertices)
##
##import sys
##a = sys.getsizeof(A)/sys.getsizeof(A2)
##
print("OK")
