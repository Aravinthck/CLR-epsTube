
import os
import sys

path = os.getcwd()
print(path)
sys.path.insert(0, path)


DATAINFO_DIR = "../DatasetsInfo"
RESULT_DIR = "../DatasetsResult"
RESULT_SYN_DIR = "../ExptsEval"


os.mkdir(DATAINFO_DIR)
os.mkdir(RESULT_DIR)
os.mkdir(RESULT_SYN_DIR)



