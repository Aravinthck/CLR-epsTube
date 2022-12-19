
from abc import ABC, abstractmethod
from Codes.utils import *



class LossOptimization(ABC):


    
    @abstractmethod
    def optimize(self, data, K ,f , *args, **kwargs):

        raise NotImplementedError



class Cl_SVR_Cg(LossOptimization):


    def __init__(self, WarmStart = True,
                outliersCnt = 0,
                initConstrCnt = 10,
                time = 1,
                optimalGap = 0.05,
                tol = 0.005,
                initConstrIntRatio = 2,
                step_plots = False,
                outputFlag = False):

        """
        A class that performs epsilon tube CLR with row generation

        Parameters
        ----------

        WarmStart :             (bool)      If we want to warm start the centers for the MILP at first iteration 
        outliersCnt:            (int)       Number of outliers (l)
        initConstrCnt:          (int)       Number of points (per cluster) for which we initialize the constraints set
        time:                   (int)       Time in minutes for which we want the MILP to run per every constraint generation iteration
        optimalGap:             (float)     Value between [0,1] to set the maximum optimality gap we expect from the MILP solver
        tol:                    (float)     An acceptable tolerance 
        bigM:                   (float)     big M value if use_bigM is "True"
        initConstrIntRatio:     (int)       Ratio that decides the number of internal points (initConstrCnt/initConstrIntRatio)
                                            per cluster for which we initialize the constraints set

        outputFlag:             (bool)      default: "False"
                                            if True, returns the progress output from the MILP solver



        
        """


        
        self.time = time
        
        self.WarmStart = WarmStart
        self.outliersCnt = outliersCnt
        self.initConstrCnt = initConstrCnt
        self.initConstrIntRatio = initConstrIntRatio
        self.optimalGap = optimalGap
        self.tol = tol
        self.step_plots = step_plots
        self.outputFlag = outputFlag


    def optimize(self, data, K, f, regularize_param , max_iter, compute, random_state,*args, **kwargs):

        """
        A function that performs the constraint generation methodology     

        """

        data = data.copy()
        X = data.to_numpy()[:,0:f]
        Y = data.to_numpy()[:,f:f+1]

        n,_ = X.shape
        
        self.randState = random_state
        self.max_iter = max_iter
        print("Clusterwise-Regression model with SVR - Constraint generation")

        print("# of outliers: ", self.outliersCnt)


        # outliers list

        trueOutliers_list = []
        outliers_list = []

        counter = 0

        # list of points that were added during the constraint generation iterations 

        addPts_list = [] 

        maxPts_list = []                # Gurobi returned max points
        trueMaxPts_list = []            # True max points excluding outliers

        # list of data with point to cluster assignment information

        df_data_list = []

        weights_list = [] 
        bias_list = [] 

        # a small epsilon value for ensuring values of centers in their first dimension are increasing 
        eps = 0.001          

        run_time = 0
        # set of points for which we generate constraints 
        cg_pts = set() 

        # Obtain the initial set of points for which we generate constraints before the first iteration of the MILP 
        
        weights, bias, assignCluster, binaryAssignVar, initConstrsEdge, initConstrsNear = initConstraints(data, K,  f, reg_param = regularize_param, addConstrs = self.initConstrCnt, randState = self.randState, ratio = self.initConstrIntRatio)

        # Initial set of constraints 
        initContrs =  initConstrsEdge.union(initConstrsNear)

        cg_pts = set(initContrs)
        n_cg_pts =  len(cg_pts)

        print("Number of WS constraints: ", n_cg_pts)
        # ensuring initial number of points for which we generate constraints is larger than the number of outliers 
        if n_cg_pts < self.outliersCnt:
            print("Add more than %0.0f intial points" %(self.outliersCnt))

        features = range(f)
        clusters = range(K) 


        # Defining the initial model in Gurobi for the first iteration in the constraint generation optimization methodology 
        
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

        # Error variable / objective value 
        error = m.addVar(vtype=gp.GRB.CONTINUOUS,lb = 0, name="E")

        # Weights and bias of the SVR and clustering
        
        w = m.addVars(clusters,features,vtype=gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name="w_kj")
        b = m.addVars(clusters,vtype=gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name="b_k")

        # the below variables need to added on the go during the iterative constraint generations 

        # Binary indicator variables  
        c = m.addVars(cg_pts, clusters, vtype=gp.GRB.BINARY, name="c_ik")


        m.modelSense = gp.GRB.MINIMIZE

        # Adding constraints to the initial Gurobi model

        # Constraints to ensure one point is assigned to only one cluster

        if self.outliersCnt > 0:

            # modified constraints to exclude exactly outliersCnt (l) number of points from being assigned to any cluster

            m.addConstrs((gp.quicksum(c[pt,k] for k in clusters) <= 1 for pt in cg_pts ),"pt_in_cluster")
            totpts = m.addConstr(gp.quicksum(c[pt,k] for k in clusters for pt in cg_pts ) ==  n_cg_pts - self.outliersCnt )
        else:
            m.addConstrs((gp.quicksum(c[pt,k] for k in clusters) == 1 for pt in cg_pts ),"pt_in_cluster")


        # Symmetry breaking constraints

        for k in range(K-1):
            m.addConstr( w[k,0] + eps <= w[k+1,0])

        # SVR regression loss function 
        m.addConstrs( ( (c[pt,k] == 1) >> (error >= (Y[pt,0] - gp.quicksum(w[k,j]*X[pt,j] for j in features) - b[k])) for pt in cg_pts for k in clusters), "abs_obj_fn1")
        m.addConstrs( ( (c[pt,k] == 1) >> (error >= -(Y[pt,0] - gp.quicksum(w[k,j]*X[pt,j] for j in features) - b[k])) for pt in cg_pts for k in clusters), "abs_obj_fn2")

        print("w at WS: ", weights)
        print("b at WS: ", bias)

        # Warm start 
        if self.WarmStart:
            # print('Warm Starting')
            for k in clusters:
                for pt in cg_pts:
                    c[pt,k].start = binaryAssignVar[pt,k]

                for j in features:
                    w[k,j].start = weights[k,j]
                
                b[k].start = bias[k]

        m.setObjective(error)

        m.setParam('TimeLimit', self.time*60) 

        m.setParam('MIPGap', self.optimalGap) 
        m.setParam('OutputFlag', self.outputFlag)

        # Constraint generation iterations 

        start = default_timer()

        while counter<self.max_iter:

            # MILP solve with Gurobi
            m.optimize()

            # Data extraction from Gurobi

            # print('\nCOST: %g' % (m.objVal))

            vals_error = error.X
        
            weights =np.zeros((K,f))

            for k in clusters:
                if k<K:
                    for j in features:
                        weights[k,j] = w[k,j].X
            bias = [x.X for x in m.getVars() if x.VarName.find('b') != -1]
    
            weights_list.append(weights)
            bias_list.append(bias)
            # print('Regression parameters \n', weights, bias)

            print("w: ", weights)
            print("b: ", bias)

            outliers = []
            cik = np.zeros((n,K))
            for pt in cg_pts:
                for k in clusters:

                    cik[pt,k] = c[pt,k].X

                if sum(cik[pt,:]) == 0:
                    outliers.append(pt)

            # Gurobi Maximum distance points 
            errWithCikSys,_ = getPredictionError(X,Y, K, weights, bias, Cik=cik)
            
            maxPts,_,_,_= getConstraintPts(errWithCikSys,K)

            maxPts_list.append(maxPts)
            
            # print('Gurobi Outliers: ', outliers)
            outliers_list.append(outliers)

            data['cluster'] = getClusterAssign(cik, outliers)

            print('E: ',vals_error)

            # Explicitly check whether the maximum distance from all points to their closest center is same as that from Gurobi
            # If not, identify the most violated constraints and add them to the constraint set and re-solve the MILP

            # Get Distance matrix

            errWithCik , trueCik =   getPredictionError(X,Y, K, weights, bias)


            # get outliers and points to add as constraints 
            ptConstrs, trueOutIndx, maxError, distMRes = getConstraintPts(errWithCik,K,self.outliersCnt)

            trueOutliers_list.append(trueOutIndx)

            trueMaxPts_list.append(ptConstrs)
            print('MaxError: ',maxError)

            # print('Adding max pts (with or without outliers) ',ptConstrs)

            # print('Adding outliers if any ', trueOutIndx)

            allAddPts = list(np.ravel(ptConstrs))
            allAddPts.extend(list(np.ravel(trueOutIndx)))
            

            # capture true labels for all points by assigning points to their closest center

            data['trueCluster'] = labels = getClusterAssign(trueCik, trueOutIndx)

            df_data_list.append(data.copy())

            if self.step_plots:
                ax = sns.scatterplot( x="X1", y="Y", data=data, hue='trueCluster', style = 'trueCluster', palette="Dark2", legend = "auto")
                # ax = sns.lineplot( x="X", y="pred", data=clus.data, hue='model',markers=False, palette="Dark2", legend = "auto")
                # plt.scatter(X[clus.outliers], Y[clus.outliers], c='C04')
                plt.scatter(data.iloc[list(initContrs),0],data.iloc[list(initContrs),f], c = 'C0', marker='v')
                # plt.scatter(data.iloc[list(cg_pts),0],data.iloc[list(cg_pts),f], c = 'C3')
                plt.scatter(data.iloc[list(allAddPts),0],data.iloc[list(allAddPts),f], c = 'C3')



                # pred = getPrediction(X,K,clus.weights,clus.bias)

                plt.show()



            # Exit here to increase maximum required iteration 
            if counter == self.max_iter-1:
                
                if any(maxError > vals_error + self.tol):

                    print('Optimal solution not reached but current cost: ', m.objVal)
                    print('Increase # of iterations')
                else:
                    print('Optimal solution reached with Cost: ', m.objVal)

                
                optval = m.objVal
                optgap = m.MIPGap
                # addPts_list.append(list(ptConstrs))

                if self.outliersCnt > 0:
                    data['trueCluster'] = labels = getClusterAssign(trueCik)


                break

            # check if we need to add any more constraints 
            if any(maxError > vals_error + self.tol):

                addptConstrs = [pt for pt in allAddPts if pt not in cg_pts]

                if not addptConstrs:
                    addptConstrs = getMoreConstraintPts(distMRes, K, cg_pts,n,trueCik)
                
                addPts_list.append(addptConstrs)

                # print("Generate constraints after repetitions are removed:" , addptConstrs)

                for pt in addptConstrs:
                    if pt not in cg_pts:
                        # print('Adding constr for: ', pt)

                        # add variables for the new points


                        for k in clusters:
                            c.update({(pt, k ): m.addVar(vtype=gp.GRB.BINARY, name= 'c_ik['+str(pt)+','+str(k)+']') })

                        # print('added vars for ', pt)

                        # add constraints for the new points

                        if self.outliersCnt > 0:
                            m.addConstr((gp.quicksum(c[pt,k] for k in clusters) <= 1  ))
                        else:
                            m.addConstr((gp.quicksum(c[pt,k] for k in clusters) == 1 ))

                        m.addConstrs( ( (c[pt,k] == 1) >> (error >= (Y[pt,0] - gp.quicksum(w[k,j]*X[pt,j] for j in features) - b[k])) for k in clusters), "abs_obj_fn1")

                        m.addConstrs( ( (c[pt,k] == 1) >> (error >= -(Y[pt,0] - gp.quicksum(w[k,j]*X[pt,j] for j in features) - b[k])) for k in clusters), "abs_obj_fn2")

                    
                    cg_pts.add(pt)

                if self.outliersCnt > 0 :
                        
                    # delete the old constraint

                    m.remove(totpts)

                    # add the new constraint under the same name
                    n_cg_pts = len(cg_pts)
                    # print("adding totpts const")
            
                    totpts = m.addConstr(gp.quicksum(c[pt,k] for k in clusters for pt in cg_pts ) ==  n_cg_pts - self.outliersCnt )

                m.update()
                    
            else:
                print('Optimal solution reached with Cost: ', m.objVal)
                optval = m.objVal
                optgap = m.MIPGap
                addPts_list.append(list(ptConstrs))

                if self.outliersCnt > 0:
                    data['trueCluster'] = labels = getClusterAssign(trueCik)

                break

            counter+=1
        
        end = default_timer()
        run_time = end - start 

        print('\n\n# of constraints added: ',len(cg_pts))

        return data, labels, trueCik, weights , bias, weights_list, bias_list, cg_pts, optval, optgap, df_data_list, initContrs, trueMaxPts_list, addPts_list, trueOutIndx, run_time






class Cl_SVR_milp(LossOptimization):


    def __init__(self, WarmStart = True,
                outliersCnt = 0,
                time = 1,
                optimalGap = 0.05,
                outputFlag = False):

        """
        A class that performs epsilon tube CLR without row generation


        
        """


        
        self.time = time       
        self.WarmStart = WarmStart
        self.outliersCnt = outliersCnt
        self.optimalGap = optimalGap
        self.outputFlag = outputFlag


    def optimize(self, data, K, f, regularize_param , compute, random_state , *args, **kwargs):

        

        data = data.copy()
        X = data.to_numpy()[:,0:f]
        Y = data.to_numpy()[:,f:f+1]

        n,_ = X.shape
        
        self.randState = random_state
        print("Clusterwise-Regression model with SVR - MILP")

        print("# of outliers: ", self.outliersCnt)

        # a small epsilon value for ensuring values of centers in their first dimension are increasing 
        eps = 0.001          

        if self.WarmStart:
            weights, bias, assignCluster, binaryAssignVar = CLR_WarmStart_Best(data, K, f, reg_param = regularize_param, randState = random_state)
            
            # print("w: ", weights)
            # print("b: ", bias)

            distManhCls, _ = getPredictionError(X,Y, K, weights, bias, Cik= binaryAssignVar)

            print("Max error for warm starting model: ", np.max(distManhCls))

        N = range(n)
        features = range(f)
        clusters = range(K) 


        # Defining the initial model in Gurobi for the first iteration in the constraint generation optimization methodology 
        
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

        # Error variable / objective value 
        error = m.addVar(vtype=gp.GRB.CONTINUOUS,lb = 0, name="E")

        # Weights and bias of the SVR and clustering
        
        w = m.addVars(clusters,features,vtype=gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name="w_kj")
        b = m.addVars(clusters,vtype=gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name="b_k")

        # the below variables need to added on the go during the iterative constraint generations 

        # Binary indicator variables  
        c = m.addVars(N, clusters, vtype=gp.GRB.BINARY, name="c_ik")


        m.modelSense = gp.GRB.MINIMIZE

        # Adding constraints to the initial Gurobi model

        # Constraints to ensure one point is assigned to only one cluster

        if self.outliersCnt > 0:

            # modified constraints to exclude exactly outliersCnt (l) number of points from being assigned to any cluster
            m.addConstrs((gp.quicksum(c[pt,k] for k in clusters) <= 1 for pt in N ),"pt_in_cluster")
            totpts = m.addConstr(gp.quicksum(c[pt,k] for k in clusters for pt in N ) ==  n_cg_pts - self.outliersCnt )
        else:
            m.addConstrs((gp.quicksum(c[pt,k] for k in clusters) == 1 for pt in N ),"pt_in_cluster")


        # Symmetry breaking constraints

        for k in range(K-1):
            m.addConstr( w[k,0] + eps <= w[k+1,0])

        # SVR regression loss function 
        m.addConstrs( ( (c[pt,k] == 1) >> (error >= (Y[pt,0] - gp.quicksum(w[k,j]*X[pt,j] for j in features) - b[k])) for pt in N for k in clusters), "abs_obj_fn1")
        m.addConstrs( ( (c[pt,k] == 1) >> (error >= -(Y[pt,0] - gp.quicksum(w[k,j]*X[pt,j] for j in features) - b[k])) for pt in N for k in clusters), "abs_obj_fn2")

        if self.WarmStart:
            print("w at WS: ", weights)
            print("b at WS: ", bias)

        # Warm start 
        if self.WarmStart:
            # print('Warm Starting')
            for k in clusters:
                for pt in N:
                    c[pt,k].start = binaryAssignVar[pt,k]

                for j in features:
                    w[k,j].start = weights[k,j]
                
                b[k].start = bias[k]

        m.setObjective(error)

        m.setParam('TimeLimit', self.time*60) 

        m.setParam('MIPGap', self.optimalGap) 
        m.setParam('OutputFlag', self.outputFlag)

        # Constraint generation iterations 
        start = default_timer()
        m.optimize()
        

        # Data extraction from Gurobi

        # print('\nCOST: %g' % (m.objVal))

        vals_error = error.X
    
        weights =np.zeros((K,f))

        for k in clusters:
            if k<K:
                for j in features:
                    weights[k,j] = w[k,j].X
        bias = [x.X for x in m.getVars() if x.VarName.find('b') != -1]

        outliers = []
        cik = np.zeros((n,K))
        for pt in N:
            for k in clusters:

                cik[pt,k] = c[pt,k].X

            if sum(cik[pt,:]) == 0:
                outliers.append(pt)

            # Gurobi Maximum distance points 
        

        data['cluster'] = data['trueCluster'] = labels = getClusterAssign(cik, outliers)

        print('E: ',vals_error)

        if self.outliersCnt > 0:
            data['trueCluster'] = labels = getClusterAssign(cik)
        

        ax = sns.scatterplot( x="X1", y="Y", data=data, hue='trueCluster', style = 'trueCluster', palette="Dark2", legend = "auto")

        plt.show()



        print('Solution reached with Cost: ', m.objVal)
        optval = m.objVal
        optgap = m.MIPGap
                


        end = default_timer()
        run_time = end - start 

        return data, labels, cik, weights , bias, 0, 0, 0, optval, optgap, 0, 0, 0, 0, outliers, run_time





class Cl_SVR_greedy(LossOptimization):


    def __init__(self, Kmeans_init = False,
                outliersCnt = 0,
                time = 1,
                optimalGap = 0.05,
                tol = 0.001,
                step_plots = False,
                outputFlag = False):

        """
                A class that performs epsilon tube CLR with em greedy method



        
        """


        
        self.time = time

        self.outliersCnt = outliersCnt
        self.Kmeans_init = Kmeans_init
        self.optimalGap = optimalGap
        self.tol = tol
        self.step_plots = step_plots
        self.outputFlag = outputFlag


    def optimize(self, data, K, f , max_iter, compute, random_state,*args, **kwargs):



        data = data.copy()
        X = data.to_numpy()[:,0:f]
        Y = data.to_numpy()[:,f:f+1]

        n,_ = X.shape
        
        self.randState = random_state
        self.max_iter = max_iter

        print("Clusterwise-Regression model with SVR - Greedy algorithm")

        print("# of outliers: ", self.outliersCnt)

        


        # outliers list

        trueOutliers_list = []
        outliers_list = []

        counter = 0
        maxError = 0
        prev_maxError = 100

        # list of data with point to cluster assignment information

        df_data_list = []

        weights_list = [] 
        bias_list = [] 

        run_time = 0
        start = default_timer()


        while (counter < self.max_iter) and ( abs(maxError - prev_maxError) > self.tol):

            # Assignment phase
            print("Iteration: ", counter)

            if counter == 0:
                data = Greedy_initialize(data, K, f , self.Kmeans_init, self.randState)

            
            else:

                # Fixing assignment for next iteration 
                prev_maxError = maxError
                errWithCik , trueCik =   getPredictionError(X,Y, K, weights, bias)
                maxError = np.max(errWithCik)
                
                print("PrevError: ", prev_maxError)
                print("MaxError: ", maxError)

                data['trueCluster'] = labels = getClusterAssign(trueCik)
                

            # Regression phase
            weights = np.zeros((K,f))
            bias = np.zeros((K,1)) 
            optval = 0
  
            for k in range(K):
                weights[k,:], bias[k],objval   = LR_with_MILP(data[data.trueCluster == k+1], f, time = self.time, optgap = self.optimalGap,  compute = compute, outputFlag=self.outputFlag)
                optval = max(optval, objval)
            print("w: ", weights)
            print("b: ", bias)

            weights_list.append(weights)
            bias_list.append(bias)

            if self.step_plots:

                ax = sns.scatterplot( x="X1", y="Y", data=data, hue='trueCluster', style = 'trueCluster', palette="Dark2", legend = "auto")

                pred = getPrediction(X,K,weights,bias)

                for k in range(K):
                    plt.plot(X[data.trueCluster == k+1,0],pred[data.trueCluster == k+1,k])
                plt.show()


            print("Iteration ends: ", counter)

            counter+=1

            if self.outliersCnt > 0:
                data['trueCluster'] = labels = getClusterAssign(trueCik)

                break

            counter+=1
        
        end = default_timer()
        run_time = end - start 

        return data, labels, trueCik, weights , bias, weights_list, bias_list, 0, optval, 0, df_data_list, 0, 0, 0, 0, run_time






class km_et(LossOptimization):


    def __init__(self, 
                outliersCnt = 0,
                time = 1,
                optimalGap = 0.05,
                tol = 0.001,
                outputFlag = False):

        """
        A class that performs epsilon tube regression after k-means


        
        """


        
        self.time = time

        self.outliersCnt = outliersCnt
        self.optimalGap = optimalGap
        self.tol = tol
        self.outputFlag = outputFlag


    def optimize(self, data, K, f , compute, random_state,*args, **kwargs):



        data = data.copy()
        X = data.to_numpy()[:,0:f]
        Y = data.to_numpy()[:,f:f+1]

        n,_ = X.shape
        
        self.randState = random_state

        print("Clusterwise-Regression model with Kmeans-et")

        print("# of outliers: ", self.outliersCnt)

        maxError = 0


        run_time = 0
        start = default_timer()


        data = Greedy_initialize(data, K, f ,True, self.randState)
        labels = data['trueCluster'] 
                

        # Regression phase
        weights = np.zeros((K,f))
        bias = np.zeros((K,1)) 
        optval = 0

        for k in range(K):
            weights[k,:], bias[k], objval   = LR_with_MILP(data[data.trueCluster == k+1], f, time = self.time, optgap = self.optimalGap,  compute = compute, outputFlag=self.outputFlag)
            optval = max(optval, objval)
        print("w: ", weights)
        print("b: ", bias)

        binaryAssignVar = np.zeros((data.shape[0], K))
        assignCluster = np.array(labels) - 1
        for i , k in enumerate(assignCluster):        
            binaryAssignVar[i,k] = 1
    

        errWithCik , trueCik =   getPredictionError(X,Y, K, weights, bias, Cik=binaryAssignVar)
        maxError = np.max(errWithCik)
                

        print("MaxError: ", optval)


        
        end = default_timer()
        run_time = end - start 

        return data, labels, trueCik, weights , bias, 0, 0, 0, optval, 0, 0, 0, 0, 0, 0, run_time






class km_lr(LossOptimization):


    def __init__(self,
                outliersCnt = 0):

        """
        A class that performs kmeans followed by lr
 
        
        """


        
        self.time = time

        self.outliersCnt = outliersCnt



    def optimize(self, data, K, f , random_state,*args, **kwargs):


        data = data.copy()
        X = data.to_numpy()[:,0:f]
        Y = data.to_numpy()[:,f:f+1]

        n,_ = X.shape
        
        self.randState = random_state

        print("Clusterwise-Regression model with KM+LR algorithm")

        print("# of outliers: ", self.outliersCnt)

        # outliers list
        # list of data with point to cluster assignment information


        run_time = 0
        start = default_timer()

        data_kmlr, km_labels , optlist_kmlr = KM_LR(data, K, f,  reg_param = 0 , randstate = self.randState)
        weights = np.zeros((K,f))
        bias = np.zeros((K,1)) 

        for i,opt in enumerate(optlist_kmlr):
            weights[i,:] = opt.coef_ 
            bias[i] = opt.intercept_     

        binaryAssignVar = np.zeros((data.shape[0], K))
        assignCluster = np.array(km_labels) - 1
        for i , k in enumerate(assignCluster):        
            binaryAssignVar[i,k] = 1
    

        errWithCik , trueCik =   getPredictionError(X,Y, K, weights, bias, Cik=binaryAssignVar)
        optval = maxError = np.max(errWithCik)
                

        print("MaxError: ", maxError)

        data['trueCluster'] = labels = km_labels
                
        
        end = default_timer()
        run_time = end - start 

        return data, labels, trueCik, weights , bias, 0, 0, 0, optval, 0, 0, 0, 0, 0, 0, run_time



class km_svr(LossOptimization):


    def __init__(self,
                epsilon = 0.1,
                outliersCnt = 0):

        """
        A class that performs SVR after kmeans
      
        """


        
        self.time = time
        self.epsilon = epsilon
        self.outliersCnt = outliersCnt



    def optimize(self, data, K, f , random_state,*args, **kwargs):


        data = data.copy()
        X = data.to_numpy()[:,0:f]
        Y = data.to_numpy()[:,f:f+1]

        n,_ = X.shape
        
        self.randState = random_state

        print("Clusterwise-Regression model with KM+SVR algorithm")

        print("# of outliers: ", self.outliersCnt)

        # outliers list
        # list of data with point to cluster assignment information


        run_time = 0
        start = default_timer()

        data_kmlr, km_labels , optlist_kmlr = KM_SVR(data, K, f,  epsilon = self.epsilon, randstate = self.randState)
        weights = np.zeros((K,f))
        bias = np.zeros((K,1)) 

        for i,opt in enumerate(optlist_kmlr):
            weights[i,:] = opt.coef_ 
            bias[i] = opt.intercept_     

        binaryAssignVar = np.zeros((data.shape[0], K))
        assignCluster = np.array(km_labels) - 1
        for i , k in enumerate(assignCluster):        
            binaryAssignVar[i,k] = 1
    

        errWithCik , trueCik =   getPredictionError(X,Y, K, weights, bias, Cik=binaryAssignVar)
        optval = maxError = np.max(errWithCik)
                

        print("MaxError: ", maxError)

        data['trueCluster'] = labels = km_labels
                
        
        end = default_timer()
        run_time = end - start 

        return data, labels, trueCik, weights , bias, 0, 0, 0, optval, 0, 0, 0, 0, 0, 0, run_time






class k_plane(LossOptimization):


    def __init__(self,
                outliersCnt = 0):

        """
        A class that performs k plane clr
  
        
        """


        
        self.time = time

        self.outliersCnt = outliersCnt



    def optimize(self, data, K, f , max_iter, random_state,*args, **kwargs):



        data = data.copy()
        X = data.to_numpy()[:,0:f]
        Y = data.to_numpy()[:,f:f+1]

        n,_ = X.shape
        
        self.randState = random_state
        self.max_iter = max_iter

        print("Clusterwise-Regression model with k-plane algorithm")

        print("# of outliers: ", self.outliersCnt)

        # outliers list
        # list of data with point to cluster assignment information


        run_time = 0
        start = default_timer()

        sc = SupervisedClustering(K=K,f=f,gmm=True, max_iter=self.max_iter, random_state = self.randState)
        sc.set_supervised_loss(LinearRegression(regularize_param = 0))
        sc.set_assignment(ArbitraryAssign()) 
        # sc.set_assignment(ClosestCentroid())

        sc.fit(data)  
        
        weights = np.zeros((K,f))
        bias = np.zeros((K,1)) 

        for i,opt in enumerate(sc.opt_list):
            try:
                weights[i,:] = opt.coef_ 
                bias[i] = opt.intercept_     
            except : 
                print("No coef for k= ", i)  

        df_data_list = sc.data_list
        binaryAssignVar = np.zeros((data.shape[0], K))
        assignCluster = sc.model - 1
        for i , k in enumerate(assignCluster):        
            binaryAssignVar[i,k] = 1
    

        errWithCik , trueCik =   getPredictionError(X,Y, K, weights, bias, Cik=binaryAssignVar)

        optval = maxError = np.max(errWithCik)
                

        print("MaxError: ", maxError)

        data['trueCluster'] = labels = getClusterAssign(trueCik)
                
        
        end = default_timer()
        run_time = end - start 

        return data, labels, trueCik, weights , bias, 0, 0, 0, optval, 0, df_data_list, 0, 0, 0, 0, run_time


