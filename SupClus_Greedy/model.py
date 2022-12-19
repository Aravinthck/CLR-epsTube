from SupClus_Greedy.utils import *
from SupClus_Greedy.loss import *
from SupClus_Greedy.assign import *

class SupervisedClustering():


    def __init__(self, K , f,
                max_iter = 10,
                gmm = False,
                KM_initialize = False,
                random_state=None
            ):
        """
        A module that combines all the components of Predictive clustering.
        """

        self.K = K
        self.f = f
        self.max_iter = max_iter
        self.gmm = gmm
        self.KM_initialize = KM_initialize

        if random_state is not None:
            self.random_state=random_state  
        else: 
            self.random_state =  np.random.randint(0, 2**16-1)



        self.param = [self.K, self.f]

        self.assign = None      # assignment object
        self.loss = None        # supervised loss object


    def parameters(self):
        return self.param

    def set_supervised_loss(self, loss):

        self.loss = loss

    def set_assignment(self, assign):

        self.assign = assign


    def fit(self, data):
        """
        
        self.assign.assign_cluster()
        self.loss.optimize_loss()

        """

        data = data.copy()

        self.data = data

        counter=0
        loss_post = list()
        loss_best = list()
        loss_pre = list()
        data_list = []
        opt_list_list = []
        eta = list()
        score_list = []

        gmm_condn = True
        loss_decr_condn = True

        while (counter < self.max_iter and ( (not self.gmm) or (gmm_condn or loss_decr_condn ))):

            # Assignment phase

            if counter == 0:

                self.data = initialize(self.data, self.K, self.f , KM_intialize = self.KM_initialize, randstate = self.random_state )
                tmp_data = self.data.copy()
            else:
                # Fixing assignment for next iteration 
                
                new_model, loss_best_pre = self.assign.assign_cluster(tmp_data, self.K, self.f)
                tmp_data['model'] = new_model
                loss_best.append(loss_best_pre)  # MSE for the best assignment - F(w_t-1)

                if self.gmm:

                    loss_pre_ = 0   # MSE for the new assignment with the previous set of weights b_t(w_t-1)
                    for i in range(self.K):
                        loss_pre_ = loss_pre_ + sum( tmp_data[tmp_data['model'] == i+1]['loss'+str(i+1)] )
                    
                    loss_pre_ = loss_pre_/tmp_data.shape[0]
                    loss_pre.append(loss_pre_)
                

   

            if counter > 1 and self.gmm:
                
                if loss_pre[counter] <= loss_post[counter-1]:
                    progress = (loss_pre[counter] - loss_post[counter-1])/(loss_best[counter] - loss_post[counter -1]+ 1e-5)
                    eta.append(progress)            
                    gmm_condn = True
                else:
                    gmm_condn = False



            # Regression/Classification phase

            tmp_data, loss_post_, score, tmp_opt_list = self.loss.optimize_loss(tmp_data, self.K, self.f)

            # print(loss_post_)

            loss_post.append(loss_post_)  #  MSE for the new assignment with new weights after minimizing the bound obtained by fixing the assignments 
            score_list.append(score)    
            
            # Loss function monotonically decreasing condition

            if counter > 0:
                if loss_post[counter] <= loss_post[counter-1]:
                    loss_decr_condn = True  
                else:
                    loss_decr_condn = False

            if counter == 0: 
                self.data = tmp_data.copy()
                data_list.append(tmp_data)
                # opt_list_list.append(opt_list)

                loss_best.append(loss_post_)
                loss_pre.append(1000) # Dummy large number

            if ( (not self.gmm) or (gmm_condn or loss_decr_condn )):
                self.data = tmp_data.copy()
                data_list.append(tmp_data)

                opt_list = tmp_opt_list
                opt_list_list.append(opt_list)

                self.loss_ = loss_post[-1]
                self.score_ = score_list[-1]
                # print('\n counter: ', counter)
                # print('loss', loss_post[-1])
                # print('condn ', gmm_condn, loss_decr_condn)
                # print(opt_list[0].coef_, opt_list[0].intercept_)
                # print(opt_list[1].coef_, opt_list[1].intercept_)
            else:
                # print('\n counter: ', counter)
                # print('loss', loss_post[-1])
                
                # print('GMM conditions are not satisfied')
                break
        

            counter+=1

        self.loss_list = loss_post
        self.model = self.data['model']
        self.data_list = data_list
        # self.Result = self.data
        self.opt_list = opt_list
        self.opt_list_list = opt_list_list
        
        return self



    def predict(self, predict_data, k_neighbors):

        # predict data has only X
        self.k_neighbors = k_neighbors 
        
        return predict_knn(predict_data, self.K, self.f, self.opt_list, self.data,self.k_neighbors)



    def validate(self, val_data, k_neighbors):

        # Validation data has both X,y
        self.k_neighbors = k_neighbors 
        self.Val_Data, self.Val_mse , self.Val_score = validate_knn(val_data, self.K, self.f, 
                self.opt_list, self.data, self.k_neighbors)
        
        return self 

    def result(self, train_no_scale):

        self.bounds = features_boundingbox(train_no_scale, self.K, self.f)
        self.centroids = features_centroid(train_no_scale, self.K, self.f)
        # self.weights = features_weights(train_no_scale, self.K, self.f, self.opt_list)

        return self   

    # def optimize(self,):

    # def predict(self):

    # def results(self):
