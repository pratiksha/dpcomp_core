from __future__ import division
from __future__ import print_function
from past.utils import old_div
from dpcomp_core.algorithm import identity
from dpcomp_core import dataset
from dpcomp_core import util
from dpcomp_core import workload
import numpy as np
'''
An example execution of one single algorithm. 
'''

epsilon = 0.1
nickname = 'ONTIME'
sample = 1E4
seed = 1
shape_list = [(5,), (10,)]
size = 1000

# Instantiate algorithm
engine = identity.identity_engine()

# Instantiate dataset
d = dataset.DatasetFromRaw(nickname=nickname, 
                           sample_to_scale=sample,
                           colnames=['DepTime'],
                           bounds=[(-100, 100, 10)])

domain_bounds = ((0, int((100 - (-100)) / 10.)),)

# Instantiate workload
w = workload.Histogram(domain_bounds=domain_bounds,
                       nbuckets_per_dim=(10,))

# Calculate noisy estimate for x
x = d.payload
x_hat = engine.Run(w, x, epsilon, seed)

# Compute error between true x and noisy estimate
diff = w.evaluate(x) - w.evaluate(x_hat)
print('Per Query Average Absolute Error:', old_div(np.linalg.norm(diff,1), float(diff.size)))
