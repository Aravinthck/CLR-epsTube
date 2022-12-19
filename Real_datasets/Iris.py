
import os
import sys
path = os.path.dirname(os.getcwd())
sys.path.insert(0, path)

from Codes.model import *
from Codes.utils import *

from scipy import stats
import pickle

RESULT_DIR = "../../DatasetsResult"
DATA_DIR = "../../Datasets"
DATAINFO_DIR = "../../DatasetsInfo"

import warnings
warnings.filterwarnings("ignore")

from sklearn import datasets


iris = datasets.load_iris()
X = iris.data  
y = iris.target
df = pd.DataFrame(np.c_[X,y], columns = ['SepalL','SepalW','PetalL', 'PetalW', 'y'])
df_data = pd.DataFrame(X, columns = ['X1','X2','X3', 'Y'])
K = 3
f = 3


# Clusterwise SVR with constraint generation 

clus = CLR(K =K , f = f, max_iter= 5, compute = False, random_state=243)
clus.set_model(Cl_SVR_Cg(outputFlag=True, initConstrCnt=5, optimalGap= 0.05, tol = 0.02, time = 2, initConstrIntRatio= 10, outliersCnt = 0, step_plots=False))

clus.fit(df_data )

print(clus.weights)
print(clus.bias)
print("Run time: ", clus.run_time)
data = df_data.to_numpy()

X = data[:,0:f]
Y = data[:,f:f+1]

ax = sns.scatterplot( x="X1", y="Y", data=clus.data, hue='trueCluster', style = 'trueCluster', palette="Dark2", legend = "auto")
# ax = sns.lineplot( x="X", y="pred", data=clus.data, hue='model',markers=False, palette="Dark2", legend = "auto")
plt.scatter(X[clus.outliers], Y[clus.outliers], c='C04')
pred = getPrediction(X,K,clus.weights,clus.bias)

for k in range(K):
    plt.plot(X[clus.binary_assign[:,k]==1,0],pred[clus.binary_assign[:,k]==1,k])
plt.show()

