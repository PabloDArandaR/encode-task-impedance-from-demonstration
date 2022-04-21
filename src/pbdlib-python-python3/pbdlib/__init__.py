from .functions import *

from .gmm import GMM
from .gmr import GMR
from .hmm import HMM
from .hsmm import HSMM
from .model import Model
from .mvn import *
from .plot import *
from .poglqr import PoGLQR, LQR, GMMLQR, LQTAugmented
from .mtmm import MTMM, VBayesianGMM, VMBayesianGMM, VBayesianHMM
from .dmp import DMP

try:
	import gui
except ImportError as e:
	print("Could not import gui: {0}".format(e.message))
	print("run : sudo apt-get install tkinter")
except:
	print("Unexpected error:", sys.exc_info()[0])
	raise

from . import utils
from . import plot

