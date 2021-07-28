#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import logging 
logging.basicConfig(level=logging.WARNING)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from hpbandster.optimizers import RandomSearch as RandomSearch
from worker_hp_TEPS import MyWorker
from algorithms import GA
from multiprocessing import Pool
from TEPS_worker import wrapper as train_function
def dict_loop(dict_, list_):
    for ID, config in enumerate(list_):
        dict_[ID] = config
    return dict_
import csv
import os

def save2csv(filename : str, dictionary :dict, first = False,iter_num : int = 0):

    #If this is a new run or file doesnt exist, overwrite/create the file
    if os.path.exists(filename) and first == False:
        write_type = 'a'
    else:
        write_type = 'w' 

    
    with open(filename, write_type) as csvfile:
        writer = csv.writer(csvfile) 
        for i in dictionary:
            writer.writerow([iter_num,i,dictionary[i]])

w = MyWorker(sleep_interval = 0, nameserver='127.0.0.1',run_id='example1')
configspace = w.get_configspace()
###Configuration Settings###
pop_size = 2
elite_size = 0.2
num_iter = 20


ga = GA(pop_size, elite_size)
population = configspace.sample_configuration(pop_size)
pop_dict = dict()
score_dict = dict()
pop_dict = dict_loop(pop_dict, population) 

config_file = "configs.csv"
score_file = "scores.csv"
first = True

for count,i in enumerate(range(num_iter)):
    pop = []
    for i in population:
        pop.append(i.get_dictionary())
    with Pool(processes = 4) as pool:
        results = pool.map(train_function, pop)
        pool.close()
        pool.join()
    scores = []
    for i in results:
        scores.append(i[0])
    score_dict = dict_loop(score_dict, scores) 
    save2csv(config_file, pop_dict,first,count)  
    save2csv(score_file, score_dict,first,count)  
    population = ga.mutate(pop_dict, score_dict)
    pop_dict = dict_loop(pop_dict, population) 
    first = False 
    for i in os.listdir("models"):
        os.remove("models/"+str(i))
                 
