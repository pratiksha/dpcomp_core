from __future__ import division
from __future__ import print_function
from past.utils import old_div
from dpcomp_core.algorithm import identity, HB, dawa
from dpcomp_core import dataset
from dpcomp_core import util
from dpcomp_core import workload

import json
import numpy as np
from timeit import default_timer as timer

from read_metadata import load_schema, load_metadata

DOUBLE_COL = 'DoubleColumnQuantization'
STR_COL = 'StringColumnQuantization'

def get_dataset(colname, nickname, quantization):
    print(quantization[colname]['type'])
    coltype = quantization[colname]['type']
    
    if coltype == DOUBLE_COL:
        min_bound = quantization[colname]['globalMin']
        max_bound = quantization[colname]['globalMax']
        granularity = quantization[colname]['granularity']
        print(min_bound, max_bound, granularity)

        # Instantiate dataset
        d = dataset.DoubleDatasetFromRaw(nickname=nickname, 
                                         colnames=[colname],
                                         bounds=[(min_bound, max_bound, granularity)])

        num_leaves = (int)((max_bound - min_bound) / float(granularity))
    elif coltype == STR_COL:
        left_bounds = quantization[colname]['leftBoundaries']
        # Instantiate dataset
        d = dataset.StringDatasetFromRaw(nickname=nickname, 
                                         colnames=[colname],
                                         left_bounds=(left_bounds,))
        num_leaves = len(left_bounds)
    else:
        raise ValueError("Invalid column type")
        
    return (d, num_leaves)

def run_engine(engine, w, x, epsilon, seed):
    start=timer()
    x_hat = engine.Run(w, x, epsilon, seed)
    
    # Compute error between true answers and noisy estimate
    diff = w.evaluate_iter(x) - w.evaluate_iter(x_hat)
    end=timer()
    time = end-start
    err = old_div(np.linalg.norm(diff,1), float(diff.size))
    print("Elapsed:", time)
    print('Per Query Average Absolute Error:', err)
    #return (time, err)
    return err

def main():
    epsilon = 1.0
    nickname = 'ONTIME'
    seeds = list(range(10))
    size = 1000

    dirname = 'ontime_private'
    schema_fname = 'short.schema'
    results_file = dirname + '_results.json'
    schema = load_schema(dirname, schema_fname)
    metadata = load_metadata(dirname)
    quantization = metadata['quantization']['quantization']

    all_results = {}
    for s in schema:
        colname = s['name']
        if colname == 'Cancelled' or colname == 'FlightDate':
            continue
        print(colname)

        (dataset, num_leaves) = get_dataset(colname, nickname, quantization)
        domain_bounds = ((0, num_leaves),)

        # Instantiate workload
        w = workload.AllBuckets(domain_bounds=domain_bounds)

        # Calculate noisy estimate for x
        x = dataset.payload

        engines = {
            'H2':HB.H2_engine(),
            'HB':HB.HB_engine(),
            'Identity':identity.identity_engine(),
            'DAWA':dawa.dawa_engine()
        }
        results = {}
        for name, e in engines.items():
            res = []
            for s in seeds:
                res.append(run_engine(e, w, x, epsilon, s))

            results[name] = [np.mean(res), np.std(res)]

        results['nqueries'] = len(w.query_list)
        results['nseeds'] = len(seeds)
        all_results[colname] = results

    with open(results_file, 'w') as f:
        f.write(json.dumps(all_results))
    
if __name__=='__main__':
    main()
