import sys
import numpy as np
import matplotlib.pyplot as plt
import os

sys.path.append(os.path.normpath(os.getcwd() + os.sep + os.pardir) + "./src/dataHandling")

import dataHandling.data_utilities as du

task_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir) + "/resources/training_data/task_1/"

list_datasets, dataset = du.combine(task_dir=task_dir, n=10)

fig_traj, axs_traj = du.plotTrajectory(list_datasets[0][:, :6])
plt.show()
