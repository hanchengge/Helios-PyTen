# pylint: disable-msg=W0614,W0401,W0611,W0622

# flake8: noqa

__docformat__ = 'restructuredtext'

# Let users know if they're missing any of our hard dependencies
hard_dependencies = ("numpy", "pandas")
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(dependency)

if missing_dependencies:
    raise ImportError("Missing required dependencies {0}".format(missing_dependencies))
del hard_dependencies, dependency, missing_dependencies

# class
#from PyTen.tenclass import tensor,tenmat,sptensor,sptenmat,ttensor
from PyTen.tenclass import *
#from PyTen.test import *
from PyTen.tools import *
from PyTen.method import tucker_als
from PyTen import Helios
#from PyTen import demo

