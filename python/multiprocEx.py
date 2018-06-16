# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 11:47:13 2018

@author: Salomon Wollenstein
"""

import multiprocessing

def worker(num):
    """thread worker function"""
    print 'Worker:', num
    return

if __name__ == '__main__':
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i,))
        jobs.append(p)
        p.start()