import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("./src/dataHandling")
import dataHandling.data_utilities as du

task_dir = "resources/training_data/task_1/"

list_datasets, dataset = du.combine(task_dir=task_dir, n = 5)

fig_traj, axs_traj = du.plotTrajectory(list_datasets[0][:,:7], "p_q")
plt.show()