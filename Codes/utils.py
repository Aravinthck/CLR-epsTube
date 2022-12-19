import gurobipy as gp
import numpy as np
import pandas as pd
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import warnings
import seaborn as sns
import random
plt.style.use('default')
from sklearn.preprocessing import StandardScaler

from collections.abc import Iterable
from scipy import stats

from timeit import default_timer
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from sklearn.cluster import KMeans
from sklearn.svm import SVR

from matplotlib.ticker import FormatStrFormatter

from SupClus_Greedy.model import *
from SupClus_Greedy.utils import *



col = [
    'tab:green',
    'tab:blue',
'tab:orange',
'tab:red',
'tab:purple',
'tab:brown',
'tab:pink',
'tab:gray',
'tab:olive',
'tab:cyan']
colors = {0:'black', 1:'magenta', 2:'green', 3:'blue',4:'black',5:'red',6:'yellow',7:'cyan',8:'olive' ,9:'brown',10:'gray'}


markers = ["D", "v", "<", "D",  "^", "s"]


def std_scale(data,f):
    data = data.copy()
    scaler = StandardScaler()
    data.iloc[:,0:f] = scaler.fit_transform(data.iloc[:,0:f])
    # self.scaler = scaler
    return data, scaler


def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el



def getPredictionError(X,Y, K, weights, bias, Cik = []):

    pred = getPrediction(X,K,weights,bias)
    
    err = abs(pred - Y.reshape(-1,1))

    trueCik = None
    if len(Cik) == 0:
        trueCik = np.zeros_like(err)
        trueCik[np.arange(len(err)), err.argmin(1)] = 1
        errWithCik = err*trueCik
    else:
        errWithCik = err*Cik
        trueCik = Cik
  
    return errWithCik, trueCik



def getPrediction(X, K, weights, bias):

    pred = np.zeros((len(X),K))
    for i in range(K):
        pred[:,i] = X @ weights[i,:] + bias[i]

    return pred 


def KM_SVR(data, K, f,  epsilon = 0 , randstate = 123):
    data = data.copy()

    kmeans = KMeans(n_clusters=K, init= 'k-means++' , random_state=randstate).fit(
        data.iloc[:, 0:f])
    kmeans_model = kmeans.labels_
    kmeans_model = list(map(lambda x: x + 1, kmeans_model))
    data['model'] =kmeans_model


    opt_list = []
    for i in range(K):

        temp_reg = SVR(kernel='linear', C = 1, epsilon=epsilon ) 
  
        opt_list.append(temp_reg)

    for i in range(K):
        if len(data[data['model'] == i+1]) > 0:
            opt_list[i].fit(data[data['model'] == i+1].iloc[:, 0:f],
                            data[data['model'] == i+1].iloc[:, f:f+1])
    
    for i in range(K):
        if (len(data[data['model'] == i+1]) > 0) & (opt_list[i] is not None):
            data.loc[data['model'] == i+1 , 'pred'] = opt_list[i].predict(data[data['model'] == i+1].iloc[:, 0:f])

    return data, kmeans_model, opt_list


def KM_LR(data, K, f,  reg_param = 0 , randstate = 123):
    data = data.copy()

    kmeans = KMeans(n_clusters=K, init= 'k-means++' , random_state=randstate).fit(
        data.iloc[:, 0:f])
    kmeans_model = kmeans.labels_
    kmeans_model = list(map(lambda x: x + 1, kmeans_model))
    data['model'] =kmeans_model


    opt_list = []
    for i in range(K):

        temp_reg = linear_model.Ridge(alpha=reg_param, fit_intercept=True, max_iter=None, normalize=False, random_state=randstate, solver='auto')
  
        opt_list.append(temp_reg)

    for i in range(K):
        if len(data[data['model'] == i+1]) > 0:
            opt_list[i].fit(data[data['model'] == i+1].iloc[:, 0:f],
                            data[data['model'] == i+1].iloc[:, f:f+1])
    
    for i in range(K):
        if (len(data[data['model'] == i+1]) > 0) & (opt_list[i] is not None):
            data.loc[data['model'] == i+1 , 'pred'] = opt_list[i].predict(data[data['model'] == i+1].iloc[:, 0:f])

    return data, kmeans_model, opt_list

def Best_CLR_Model(data, K, f, reg_param = 0,  randState= 123):

    data = data.copy()

    obj_list = []
    
    # assign_list = [ArbitraryAssign(),  ClosestCentroid(), BoundingBox()]
    # assign_list = [ArbitraryAssign(), ArbitraryAssign(), ClosestCentroid(), ClosestCentroid(), ClosestCentroid()]
    # assign_list = [ArbitraryAssign(), ArbitraryAssign()  , ClosestCentroid(),ClosestCentroid()]
    assign_list = []

    mae_list = []

    for clus in assign_list:

        sc = SupervisedClustering(K=K,f=f,gmm=True, max_iter=20, random_state = randState)
        sc.set_supervised_loss(LinearRegression(regularize_param = reg_param))
        sc.set_assignment(clus)
        sc.fit(data)  

        obj_list.append(sc)

        mae_list.append(mae(sc.data.Y, sc.data.pred))

        if mae_list[-1] == 0:
          print("got zero objective")
          break
        else:
          print("greedy objective: ", mae_list[-1])

    data_kmlr, km_labels , optlist_kmlr = KM_LR(data, K, f,  reg_param = reg_param , randstate = randState)

    mae_list.append(mae(data_kmlr.Y, data_kmlr.pred))

    if mae_list[-1] == 0:
        print("got zero objective")
    else:
        print("greedy objective kmlr: ", mae_list[-1])

    bestmodel = np.argmin(mae_list)

    # print("Best warm start with: ", assign_list[bestmodel])

    weights = np.zeros((K,f))
    bias = np.zeros((K,1)) 

    if bestmodel == len(assign_list):
        bestmodelopt_list = optlist_kmlr
        bestLabels = km_labels
        print("Best WS is kmlr")
    else:
        bestmodelObj = obj_list[bestmodel]
        bestLabels =  bestmodelObj.model
        bestmodelopt_list = bestmodelObj.opt_list
    
    for i,opt in enumerate(bestmodelopt_list):
        try:
            weights[i,:] = opt.coef_ 
            bias[i] = opt.intercept_     
        except : 
            print("No coef for k= ", i)
    # print(weights)
    # print(bias)

    return weights, bias, bestLabels


def CLR_WarmStart_Best(data, K, f, reg_param = 0, randState= 123 ):
    
    data = data.copy()

    weights, bias, bestLabels = Best_CLR_Model(data, K, f, reg_param = reg_param, randState= randState)
    
    # print(weights)
    # print(bias)
    # print(bestLabels)

    w_sort = list(weights[:, 0].argsort())
    dictCluster = { j:i for i,j in enumerate(w_sort) }
    # dictCluster = dict(enumerate(weights[:, 0].argsort()))
    weights = weights[w_sort]
    bias = bias[w_sort] 
    # print(bestLabels[0:5])
    assignCluster = [dictCluster.get(x-1) for x in bestLabels ]
    # print(assignCluster[0:5])
    
    binaryAssignVar = np.zeros((data.shape[0], K))

    for i , k in enumerate(assignCluster):        
        binaryAssignVar[i,k] = 1
    
    return weights, bias, assignCluster, binaryAssignVar


def initConstraints(data, K,  f, reg_param = 0, addConstrs = 5, randState = 123, ratio = 2):   
    data = data.copy()
    initConstrsEdge = set()
    initConstrsNear = set()

    XX = data.to_numpy()
    X = XX[:,0:f]
    Y = XX[:,f:f+1]

    weights, bias, assignCluster, binaryAssignVar = CLR_WarmStart_Best(data, K, f, reg_param = reg_param, randState = randState)
    
    # print("w: ", weights)
    # print("b: ", bias)
    distManhCls, _ = getPredictionError(X,Y, K, weights, bias, Cik= binaryAssignVar)

    print("Max error for warm starting model: ", np.max(distManhCls))
    
    argSortDist = np.argsort(distManhCls, axis = 0)

    minIndx = np.sum(binaryAssignVar==0,axis = 0)
    initConstrsEdge = set(argSortDist[-addConstrs:].flatten())

    for k in range(K):
        initConstrsNear.update(set(argSortDist[minIndx[k]:minIndx[k]+int(addConstrs//ratio)][:,k].flatten()))

    return weights, bias, assignCluster, binaryAssignVar, initConstrsEdge, initConstrsNear

    

def getDistAssignMat(X, center, Cik = []):

    dist = manhattan_distances(X,center)
    trueCik = None
    if len(Cik) == 0:
        trueCik = np.zeros_like(dist)
        trueCik[np.arange(len(dist)), dist.argmin(1)] = 1
        distM = dist*trueCik
    else:
        distM = dist*Cik
        

    return distM, trueCik



def getClusterAssign(Cik, outliers = []):

    ClusterAssign = Cik.argmax(axis = 1) + 1

    if len(outliers) !=0:
        ClusterAssign[outliers] = 0
        
    return ClusterAssign 



def getConstraintPts(distM,K,outliersCnt = 0 ):

    trueOutIndx = []

    if outliersCnt>0:
        
        distMO = distM.copy()
        # outIndxFlat = np.argsort(distM.flatten())[-(outliersCnt+K):]
        outIndxFlat = np.argpartition(distM.flatten(), -outliersCnt)[-outliersCnt:] 

        # outThres = distM.flatten()[outIndxFlat[K]]
        outThres = distM.flatten()[outIndxFlat[0]]

        # print('True Outliers threshold: ', outThres)

        trueOutIndx = np.floor(outIndxFlat/K).astype(int)

        # print('True outliers: ', trueOutIndx )

        # trueoutliers_list.append(trueOutIndx[-outliersCnt:] )

        distMO[distMO >= outThres] = 0

        # max_error = np.max(distMO, axis=0)
        # print("max error excluding outliers: ", (max_error) )

        maxDistPtsK = list(np.argmax(distMO,axis = 0))

        # print('actual max pts when outliers',maxDistPtsK)

        maxError = np.array([distMO[j,i] for i , j in enumerate(maxDistPtsK)])


        # print("max error excluding outliers (argmax): ", (maxError) )


    else:

        maxDistPtsK = list(np.argmax(distM,axis = 0))
        maxError = np.array([distM[j,i] for i , j in enumerate(maxDistPtsK)])
        # print('actual max pts without outliers',maxDistPtsK)
        # print("max error excluding outliers (argmax): ", (maxError) )
        distMO = distM

    
    return maxDistPtsK, trueOutIndx, maxError, distMO

 
def getMoreConstraintPts(distM, K,cg_pts,n,trueCik):

    ptConstrs = []
    argSort = np.argsort(distM, axis = 0)

    for k in range(K):
        addpts_k = []
        argMaxIndx = -2 
        add_pt = argSort[argMaxIndx][k]

        while add_pt in cg_pts and argMaxIndx>(-n+1): 
            argMaxIndx-=1
            add_pt = argSort[argMaxIndx][k]
        addpts_k.extend([add_pt])


        argMinIndx = sum(trueCik[:,k]==0)
        if argMinIndx == len(trueCik):
            # all values in the column are zezo, can't find the closest pts to center
            continue
        add_pt = argSort[argMinIndx][k]

        while add_pt in cg_pts and argMinIndx<n-1:
            argMinIndx+=1
            add_pt = argSort[argMinIndx][k]
            
            # print('add pts', add_pt)
        addpts_k.extend([add_pt])

        # print('Max Min', k, addpts_k)

        ptConstrs.extend(addpts_k)    

    # print('Add more point constraints: ', ptConstrs)

    return ptConstrs






def Greedy_initialize(data, K, f, k_means = False, seed = 123):
    data = data.copy()

    if not k_means:

        generator = np.random.RandomState(seed)
        data['trueCluster'] = generator.randint(1, high = K+1, size=data.shape[0])

    else:    
        kmeans = KMeans(n_clusters=K, init= 'k-means++' , random_state=seed).fit(
            data.iloc[:, 0:f])
        kmeans_model = kmeans.labels_
        kmeans_model = list(map(lambda x: x + 1, kmeans_model))
        data['trueCluster'] =kmeans_model

    return data 


def LR_with_MILP(data, f, time = 0.5, optgap = 0.1,  compute = False, outputFlag = False):
    data = data.copy()
    data = data.to_numpy()

    # model parameters
    X = data[:,0:f]
    Y = data[:,f]

    n,f = X.shape

    outliers = 0.0


    N = range(n)
    features = range(f)

    if compute:
        e = gp.Env(empty=True)
        e.setParam('WLSACCESSID', 'fb09d92f-0384-40fd-b3b5-653ec0250044')
        e.setParam('WLSSECRET', '545f8e3f-d661-4850-80ee-3aec7e4329be')
        e.setParam('LICENSEID', 874321)
        e.start()

        # Create the model within the Gurobi environment
        m = gp.Model(env=e)    
    else:
        m = gp.Model()


    error = m.addVar(vtype=gp.GRB.CONTINUOUS,lb = 0, name="E")

    w = m.addVars(features,vtype=gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name="w_kj")
    b = m.addVar(vtype=gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name="b")

    m.modelSense = gp.GRB.MINIMIZE


    # Cost function

    m.addConstrs( ( (error >= (Y[i] - gp.quicksum(w[j]*X[i,j] for j in features) - b)) for i in N ), "abs_obj_fn1")

    m.addConstrs( ( (error >= -(Y[i] - gp.quicksum(w[j]*X[i,j] for j in features) - b)) for i in N), "abs_obj_fn2")



    m.setObjective(error)

    m.setParam('TimeLimit', time*60) 

    m.setParam('MIPGap', optgap) 
    m.setParam('OutputFlag', outputFlag)

    m.optimize()

    # print('\nCOST: %g' % (m.objVal))

    model_cluster = np.zeros(len(X))


    weights =np.zeros((f,1))

    for j in features:

        weights[j] = w[j].X

    bias = [x.X for x in m.getVars() if x.VarName.find('b') != -1]

    return weights.reshape((1,-1)), bias, m.objVal

