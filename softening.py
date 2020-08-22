from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import json

class CBPolicy(object):
    def __init__(self, act_num):
        self.act_num=act_num
        
    def friendly_soften(self, action, soften_params):
        alpha = soften_params['alpha']
        beta = soften_params['beta']
        
        soft_action = np.copy(action)
        
        probs = np.zeros(len(action)) #p
        probas = np.ones((len(action), self.act_num)) #probas
        
        for i in range(len(action)):
            u = np.random.uniform(-0.5, 0.5)
            explore_prob = alpha + beta * u
            
            probas[i] = (1-explore_prob) / (self.act_num - 1) * probas[i]
            
            probs[i] = probas[i][soft_action[i]] = explore_prob
            
            #probas[i] = probas[i] / np.sum(probas[i])
            
            #probs[i]=probas[i][soft_action[i]] #explore_prob
            
            if np.random.uniform(0,1) > explore_prob:
                a_soft = int(np.random.randint(self.act_num, size=1))
                
                while(a_soft == action[i]):
                    a_soft = int(np.random.randint(self.act_num, size=1))
                    
                soft_action[i] = a_soft
                probs[i] = probas[i][soft_action[i]]
                #probs_lists[i][soft_action[i]]=   (1-explore_prob)/(self.act_num-1)
        return soft_action, probs, probas
    
    def adversarial_soften(self, action, soften_params):
        alpha = soften_params['alpha']
        beta = soften_params['beta']
        
        soft_action = np.copy(action)
        probs = np.zeros(len(action))
        
        probas = np.ones((len(action), self.act_num))
        
        for i in range(len(action)):
            u = np.random.uniform(-0.5, 0.5)
            explore_prob = alpha+beta*u
            
            probas[i] = (1-explore_prob) / (self.act_num) * probas[i]
            a_soft = int(np.random.randint(self.act_num, size=1))
            
            while(a_soft==soft_action[i]):
                a_soft = int(np.random.randint(self.act_num, size=1))
            
            soft_action[i]=a_soft
            probas[i] = probas[i]/np.sum(probas[i])
            probs[i] = probas[i][soft_action[i]] = explore_prob # explore_prob

            if np.random.uniform(0, 1) > explore_prob:
                soft_action[i] = int(np.random.randint(self.act_num, size=1))
                probs[i]=probas[i][soft_action[i]] #(1-explore_prob)/(self.act_num)
                
        return soft_action, probs, probas
    
    def neutral_soften(self, action, soften_params):
        soft_action = np.random.randint(self.act_num, size=len(action))        
        probs = np.ones(len(action))*1/self.act_num
        #probs=probs/sum(probs)
        
        return soft_action, probs,  np.ones((len(action), self.act_num))*1/self.act_num
    
    def get_soften_action(self, action, soften_params):
        if soften_params != None:
            soft_action = []
            if soften_params['method'] == "friendly":
                soft_action, probs, prob_list = self.friendly_soften(action, soften_params)
            elif soften_params['method'] == "adversarial":
                soft_action, probs, prob_list = self.adversarial_soften(action, soften_params)
            elif soften_params['method'] == "neutral":
                soft_action, probs, prob_list = self.neutral_soften(action, soften_params)
            return {"action": [soft_action], "prob": probs, "prob_list": prob_list.tolist()}
        else:
            probs = []
            return {"action": action, "prob": probs, "prob_list": probs}
        

def train_target(X, Y, test_size, clf):
    """
        Trains a classifier and returns training model accuracy, the number of actions, the actual actions from the test data,         deterministic actions and the testing data X.

        Parameters
        ----------
        X : Pandas Dataframe
            Covariate data
        Y : str
            Target Data
        test_size : float
            Testing proportion from data
        clf : sklearn model
            sklearn model
            
        Returns
        -------
        acc: float
            Model Accuracy
        num_actions:
            Number of actions equal to the number of classes
        act_gtruth: numpy array
            The actual actions from training data
        act_pred:
            the actions predicted from classifer(deterministic policy)
        X_test:
            The covariate proportion in testing data
        """
    Y_cat_map = dict( enumerate( Y['Target'].cat.categories ) )
    Y_cat_inv_map = {v: k for k, v in Y_cat_map.items()}
    num_actions = len(Y_cat_map)
    
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        Y, 
                                                        test_size=test_size,
                                                        random_state=1)

    clf.fit(X_train, np.ravel(y_train))
    acc=clf.score(X_test, y_test) #Compute accuracy
    
    y_pred = clf.predict(X_test) #Compute the y_pred
    
    #creating action arrays
    act_pred = np.array((pd.Series(y_pred)).map(Y_cat_inv_map))
    act_gtruth = np.array((pd.Series(y_test.Target)).map(Y_cat_inv_map))
    
    return acc, num_actions, act_gtruth, act_pred, X_test.reset_index(drop=True)

def get_reward(act_pred, act_test):
    """
        Compares two lists or arrays and returns an array containing rewards

        Parameters
        ----------
        act_pred : array(int)
            Action array predicted from softened policy
        act_test : array(int)
            Action array from the ground policy

        Returns
        -------
        array(int) : array containing rewards
        """
    return np.array((np.array(act_pred)==np.array(act_test)), dtype='int64')

def classification_to_cb_behaviour(X_context, action_num, act_gtruth, act_deterministic, cb_policy, soften_params, out_file_json, N=5000):
    """
        Trains a deterministic policy, softens and creates a cb dataset from a classification dataset

        Parameters
        ----------
        X : Pandas Dataframe
            Covariate data
        Y : Pandas Dataframe
            Target Data
        test_size : float
            Testing proportion from data
        clf : sklearn model
            sklearn model
        soften_params:dict
            Softening parameters
        out_behaviour_json:file path
            A file path to the generated data for behaviour policy
        out_evaluation_json:file path
            A file path to the generated data for evaluation policy
            
        Returns
        -------
        df : Pandas Dataframe
            Covariate data to use for evaluation policy
            Number of actions
        act_det: list
            Actions from the deterministic policy (totalling to N datapoints to use in evaluation policy)
        act_gtruth: list
            Actual actions in the data (totalling to N datapoints to use in evaluation policy)
        """
    num_a = action_num*(np.ones(N, dtype='int64'))
    #print("Deterministic policy ground truth accuracy(G_truth reward): ", accuracy)
   
    df = pd.DataFrame()
    r=np.zeros(N)
    a_det=np.zeros(N)
    a_gtruth=np.zeros(N)
    a=np.zeros(N)
    p=np.zeros(N)
    probas=np.zeros((N, action_num))
    
    for i in range(N):
        idx=np.random.randint(X_context.shape[0])
        df=df.append(X_context.iloc[[idx]])
        a_det[i]=act_deterministic[idx]
        a_gtruth[i]=act_gtruth[idx]
        act_prob=cb_policy.get_soften_action([act_deterministic[idx]], soften_params)
        r[i]=get_reward(act_gtruth[idx], act_prob['action'][0])[0]
        a[i]=act_prob['action'][0][0]
        p[i]=act_prob['prob'][0]
        probas[i]=act_prob['prob_list'][0]
            
    df=df.reset_index(drop=True)
    class2cb_json(df, a.tolist(), r.tolist(), p.tolist(), probas.tolist(), num_a.tolist(), out_file_json)
    #print("action_num behaviour: ", action_num)
    return df, a_det, a_gtruth

def classification_to_cb_evaluation(X_context, action_num, act_gtruth, act_deterministic, cb_policy, soften_params, out_file_json, N=5000):
    """
        Trains a deterministic policy, softens and creates a cb dataset from a classification dataset

        Parameters
        ----------
        X_context : Pandas Dataframe
            Covariate data
        action_num : int
            Number of actions
        act_deterministic: list
            Actions from the deterministic policy
        act_gtruth: list
            Actual actions in the data 
        cb_policy : CBPolicy
            CBPolicy object
        clf : sklearn model
            sklearn model
        soften_params:dict
            Softening parameters
        out_file_json:file path
            A file path to the generated dat
        N: int
            Number of data points
        Returns
        -------
        None
        """
    assert N==X_context.shape[0]
    num_a = action_num*(np.ones(N, dtype='int64'))
    #print("Deterministic policy ground truth accuracy(G_truth reward): ", accuracy)
    
    #cb_policy = CBPolicy(action_num)

    act_prob=cb_policy.get_soften_action(np.array(act_deterministic, dtype='int64'), soften_params)
    r=get_reward(np.array(act_gtruth, dtype='int64'), act_prob['action'][0])
    a=np.array(act_prob['action'][0], dtype='int64')
    p=act_prob['prob']
    probas=act_prob['prob_list']
    #print("Action Num", action_num)
    class2cb_json(X_context, a.tolist(), r.tolist(), p.tolist(), probas, num_a.tolist(), out_file_json)
    return r
    
def class2cb_json(X_df, a, r, p, probas, num_a, out_file_json):
    """
        Formatting function from Pandas Dataframes to write to a JSON text file

        Parameters
        ----------
        X_df : Pandas Dataframesoftened 
            Covariate data
        a : list or np array
            Actions from behaviour policy
        r : list or np array
            rewards from behaviour policy
        p : list or np array
            probs from behaviour policy
        num_a : list or np array
            number of actions
        out_file_json:file path
            A file path to the generated data
            
        Returns
        -------
        None
        """
    context_list=[]
    context={}
    for i, row in X_df.iterrows():
        #context="{"
        context={}
        for j, column in row.iteritems():
            context[j]=column
        context_list.append(json.dumps(context))
    
    parsed_data = pd.DataFrame(list(zip(context_list, a, r, p, probas, num_a)), 
               columns =['XNamespace', 'a', 'r', 'p', 'probas', 'num_a'])
    
    parsed_data_list=[]
    for i, row in parsed_data.iterrows():
        context={}
        for j, column in row.iteritems():
            context[j]=column
        parsed_data_list.append(json.dumps(context))
        
    with open (out_file_json,"w")as fp:
        for line in parsed_data_list:
            #y = json.dumps(x)
            fp.write(line+"\n")
    fp.close()