import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("./src/dataHandling")
import dataHandling.data_utilities as du

task_dir = "resources/training_data/task_1/"

datasets = du.loadTaskSets(task_dir)
datasets = list(filter(lambda x: du.filterAmount(x, 10), datasets))
datasets = list(map(du.eliminateTimeOffset, datasets))
dataset = du.combineData(datasets)