import os
import sys
import os
import numpy as np

sys.path.append("./src/dataHandling/")
import data_utilities

if len(sys.argv) <= 1:
    print("ERROR: No argument given.")
elif os.path.isdir(sys.argv[1]):
    data = data_utilities.deleteLastIter(np.load(sys.argv[1]))
    np.save(sys.argv[1], data)
else:
    print("ERROR: Path does not exist.")