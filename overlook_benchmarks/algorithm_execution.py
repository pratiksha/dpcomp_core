from __future__ import division
from __future__ import print_function
from past.utils import old_div
from dpcomp_core.algorithm import identity, HB, dawa
from dpcomp_core import dataset
from dpcomp_core import util
from dpcomp_core import workload

import numpy as np
from timeit import default_timer as timer

epsilon = 0.1
nickname = 'ONTIME'
sample = 1E4
seed = 1
size = 1000

# Instantiate algorithm
engine1 = HB.H2_engine()
engine2 = identity.identity_engine()
engine3 = dawa.dawa_engine()

# Instantiate dataset
d = dataset.DatasetFromRaw(nickname=nickname, 
                           sample_to_scale=sample,
                           colnames=['DepTime'],
                           bounds=[(-100, 100, 10)])

domain_bounds = ((0, int((100 - (-100)) / 10.)),)

# Instantiate workload
w = workload.AllBuckets(domain_bounds=domain_bounds)

# Calculate noisy estimate for x
x = d.payload
start = timer()
x_hat = engine1.Run(w, x, epsilon, seed)

# Compute error between true answers and noisy estimate
diff = w.evaluate(x) - w.evaluate(x_hat)
end = timer()
print("Elapsed:", end-start)
print('Per Query Average Absolute Error:', old_div(np.linalg.norm(diff,1), float(diff.size)))

start = timer()
x_hat = engine2.Run(w, x, epsilon, seed)

# Compute error between true answers and noisy estimate
diff = w.evaluate(x) - w.evaluate(x_hat)
end = timer()
print("Elapsed:", end-start)
print('Per Query Average Absolute Error:', old_div(np.linalg.norm(diff,1), float(diff.size)))

start=timer()
x_hat = engine3.Run(w, x, epsilon, seed)

# Compute error between true answers and noisy estimate
diff = w.evaluate(x) - w.evaluate(x_hat)
end=timer()
print("Elapsed:", end-start)
print('Per Query Average Absolute Error:', old_div(np.linalg.norm(diff,1), float(diff.size)))
