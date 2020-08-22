from vowpalwabbit import pyvw

import cressieread
import ips_snips
import mle
import ds_parse
import dr
import policy

import ast
import numpy as np
import pandas as pd
import random

from sklearn.metrics import mean_squared_error
from sklearn import linear_model

def compute_estimate(pol, est, data_file, evaluation_data=None):
    """
        Given data logs collected by using a logging policy, computes the expected reward for a provided target policy using an         estimator provided

        Parameters
        ----------
        pol : str
            target policy to evaluate
        est : str
            Estimator to use
        data_file : file
            Logged behavioural data
        evaluation_data : file
            Logged evaluation data

        Returns
        -------
        estimate: list(float)
            The computed expected rewards
        """

    a_pred = 1 
    p_pred=0

    num_samples = 0
    estimates = []
    logging_rewards = []
    
    pol_type=None
    cb_vw=None
    policy_pi=None
    vw_string=None
    evaluation_data=None
    
    behaviour_data=open(data_file,'r').readlines()
    
    if isinstance(pol, dict):
        if 'CustomVW' in pol:
            vw_string=pol['CustomVW']
            pol_type='CustomVW'
        if 'CustomCB' in pol:
            evaluation_data = open(pol['CustomCB'],'r').readlines()
            pol_type='CustomCB'
            
    if isinstance(pol, str):
        pol_type=pol
        
    if vw_string is not None:
        cb_vw=pyvw.vw(vw_string)
        policy_pi = policy.VWPolicy(cb_vw)

    # Init estimators
    if(est=='ips' or est=='snips'):
        estimator = ips_snips.Estimator()
    else:
        estimator = eval(est).Estimator()

    for i, data in enumerate(behaviour_data):
        data = ast.literal_eval(data)
    
        if pol_type=='CustomCB':
            probas = ast.literal_eval(evaluation_data[i])['probas']
            p_pred = ast.literal_eval(evaluation_data[i])['p']
            
        else:
            if vw_string is None:
                probas, _, p_pred =  compute_p_pred(pol_type, data)
            else:
                probas, _, p_pred =  compute_p_pred(pol_type, data, policy_pi=policy_pi)

        # Update estimators with tuple (p_log, r, p_pred)
        estimator.add_example(data['p'], data['r'], p_pred)

        logging_rewards.append(float(data['r']))

        if(est=='ips' or est=='snips'):
            estimates.append(estimator.get_estimate(est))
        else:
            estimates.append(estimator.get_estimate())

    return estimates

def compute_estimate_dr(pol, est, data_file, scikit=False, clf=None, evaluation_data=None, load_pretrained_model=False):
    """
        Given data logs collected by using a logging policy, computes the expected reward for a provided target policy using an         estimator provided. Computes for doubly-roboust based estimators

        Parameters
        ----------
        pol : str
            target policy to evaluate
        est : str
            Estimator to use
        data_file : file
            Logged data
        scikit: bool
            flag, if true then train using scikit, otherwise use vw
        evaluation_data: dict
            for CustomCB policies, this is the data used in the evaluation policy
        clf: linear_model
            if scikit is used, this is a model instantiated from the linear_model module
        load_pretrained_model: bool
            flag, if true then do not train model afresh, otherwise train afresh
        Returns
        -------
        
        estimates: list
            The list of computed expected rewards for all the samples
        
        """


    if not scikit:
        estimates = []

        reward_estimator_model = pyvw.vw(quiet=True)

        est, mode = dr_mode(est)

        # Init estimators
        estimator = eval(est).Estimator()

        train_ratio = 0.5
        train_size = int(sum(1 for line in open(data_file)) * train_ratio)

        idxx = 0
        train_data_batch = []
        train_data_batch_str = ""

        pol_type=None
        cb_vw=None
        policy_pi=None
        vw_string=None
        evaluation_data=None
        probas=None
        p_pred=None
    
        behaviour_data=open(data_file,'r').readlines()

        if isinstance(pol, dict):
            if 'CustomVW' in pol:
                vw_string=pol['CustomVW']
                pol_type='CustomVW'
            if 'CustomCB' in pol:
                evaluation_data = open(pol['CustomCB'],'r').readlines()
                pol_type='CustomCB'
        
        if isinstance(pol, str):
            pol_type=pol
        
        if vw_string is not None:
            cb_vw=pyvw.vw(vw_string)
            policy_pi = policy.VWPolicy(cb_vw)
       
        for i, data in enumerate(behaviour_data):
            data = ast.literal_eval(data)
            
            if pol_type=='CustomCB':
                probas = ast.literal_eval(evaluation_data[i])['probas']
                p_pred = ast.literal_eval(evaluation_data[i])['p']
            else:
                if vw_string is None:
                    probas, _, p_pred =  compute_p_pred(pol_type, data)
                else:
                    probas, _, p_pred =  compute_p_pred(pol_type, data, policy_pi=policy_pi)

            #logging_rewards.append(float(data['r']))
            if(mode == 'online'):
                #construct training example (x, a, r)
                train_example = str(data['r']) + " |ANamespace Action:" + str(data['a']) + " |XNamespace " + str(data['XNamespace'])
                #print(train_example)

                #At the beginning, you don't have a model to predict with, so you can only learn a reward estimator
                if(idxx==0):
                    reward_estimator_model = estimator.reward_estimator(train_example, reward_estimator_model, 'online', load_pretrained_model)
                    estimates.append(0)

                #Now you have a reward estimator to predict with
                elif(idxx>0):
                    x = " |XNamespace " + str(data['XNamespace'])

                    #compute estimate
                    estimator.add_example(x, data['a'], data['a_vec'], data['p'], data['r'], p_pred, probas, reward_estimator_model)
                    estimates.append(estimator.get_estimate())

                    #Update reward estimator
                    reward_estimator_model = estimator.reward_estimator(train_example, reward_estimator_model, 'online', load_pretrained_model)
            elif(mode == 'seq_online'):
                pass
            
            elif(mode == 'batch'):
                #first half, train on second half data
                if(idxx < train_size):
                    if(idxx == 0):
                        for tr_exs in behaviour_data[train_size:]:
                            tr_exs = ast.literal_eval(tr_exs)
                            train_example=str(tr_exs['r']) + " |ANamespace Action:" + str(tr_exs['a']) + " |XNamespace " + str(tr_exs['XNamespace'])
                            train_data_batch.append(train_example)
                            
                        reward_estimator_model = estimator.reward_estimator(train_data_batch, reward_estimator_model, 'batch')
                    
                    #use reward estimator to compute estimate
                    x = " |XNamespace " + str(data['XNamespace'])
                    estimator.add_example(x, data['a'], data['a_vec'], data['p'], data['r'], p_pred, probas, reward_estimator_model)
                    estimates.append(estimator.get_estimate())
                    
                #second half, train on first half data
                else:
                    if(idxx == train_size):
                        reward_estimator_model = pyvw.vw(quiet=True)
                        train_data_batch=[]
                        for tr_exs in behaviour_data[0:train_size]:
                            tr_exs = ast.literal_eval(tr_exs)
                            train_example=str(tr_exs['r']) + " |ANamespace Action:" + str(tr_exs['a']) + " |XNamespace " + str(tr_exs['XNamespace'])
                            train_data_batch.append(train_example)
                            
                        reward_estimator_model = estimator.reward_estimator(train_data_batch, reward_estimator_model, 'batch')
                    
                    x = " |XNamespace " + str(data['XNamespace'])
                    estimator.add_example(x, data['a'], data['a_vec'], data['p'], data['r'], p_pred, probas, reward_estimator_model)
                    estimates.append(estimator.get_estimate())
            
            elif(mode == 'seq_batch'):
                pass
            
            idxx += 1

        return estimates

    else:
        return compute_estimate_dr_scikit(pol, est, data_file, clf, load_pretrained_model=False)


def compute_estimate_dr_scikit(pol, est, train_data, clf=None, behaviour_data=None, load_pretrained_model=False):
    """
        Given data logs collected by using a logging policy, computes the expected reward for a provided target policy using an         estimator provided. Computes for doubly-roboust based estimators

        Parameters
        ----------
        pol : str
            target policy to evaluate
        est : str
            Estimator to use
        train_data : pd.DataFrame
            Logged data
        clf: linear_model
            if scikit is used, this is a model instantiated from the linear_model module
        load_pretrained_model: bool
            flag, if true then do not train model afresh, otherwise train afresh
        Returns
        -------
        
        estimates: list
            The list of computed expected rewards for all the samples
        
        """

    estimates = []

    reward_estimator_model = pyvw.vw(quiet=True)

    if clf is None:
        clf=linear_model.SGDClassifier(max_iter = 1000, tol=1e-3, penalty = "elasticnet")

    est, mode = dr_mode(est)

    # Init estimators
    estimator = eval(est).Estimator()

    train_ratio = 0.5
    train_size = int(train_data.shape[0] * train_ratio)

    idxx = 0
    train_data_batch = []

    data_batch = pd.DataFrame(columns =list(train_data.columns))
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    classes=np.unique(train_data[['r']])
    
    pol_type=None
    cb_vw=None
    policy_pi=None
    vw_string=None
    evaluation_data=None

    if isinstance(pol, str):
        pol_type=pol
            
    if vw_string is not None:
        cb_vw=pyvw.vw(vw_string)
        policy_pi = policy.VWPolicy(cb_vw)

    for i in range(0, train_data.shape[0]):
        data = train_data.iloc[[i]]
        if vw_string is None:
            probas, _, p_pred =  compute_p_pred(pol_type, data)
        else:
            probas, _, p_pred =  compute_p_pred(pol_type, data, policy_pi=policy_pi)

        if(mode == 'online'):
            #At the beginning, you don't have a model to predict with, so you can only learn a reward estimator
            if(idxx==0):
                reward_estimator_model = estimator.reward_estimator_scikit(data, clf, classes, 'online', load_pretrained_model)
                estimates.append(0)

            #Now you have a reward estimator to predict with
            elif(idxx>0):
                x = data[['c_time_of_day','c_user']]

                #compute estimate
                estimator.add_example_scikit(x, int(data['a']), list(map(int, list(data['a_vec'])[0])), float(data['p']), int(data['r']), p_pred, probas, reward_estimator_model)
                estimates.append(estimator.get_estimate())

                #Update reward estimator
                reward_estimator_model = estimator.reward_estimator_scikit(data, clf, classes, 'online',  load_pretrained_model)

        elif(mode == 'batch'):
            
            if(idxx < train_size):
                if(idxx == 0):
                    for j in range(train_size, train_data.shape[0]):
                        tr_exs = train_data.iloc[[j]]
                        data_batch = data_batch.append(tr_exs)
                    reward_estimator_model = estimator.reward_estimator_scikit(data_batch, clf, classes, 'batch',  load_pretrained_model)
                #use reward estimator to compute estimate
                x = data[['c_time_of_day','c_user']]

                #compute estimate
                estimator.add_example_scikit(x, int(data['a']), list(map(int, list(data['a_vec'])[0])) , float(data['p']), int(data['r']), p_pred, probas, reward_estimator_model)

                estimates.append(estimator.get_estimate())
                            
            #use batch training examples to train a reward estimator
            else:
                if(idxx == train_size):
                    for j in range(0, train_size):
                        tr_exs = train_data.iloc[[j]]
                        data_batch = data_batch.append(tr_exs)
                    reward_estimator_model = estimator.reward_estimator_scikit(data_batch, clf, classes, 'batch',  load_pretrained_model)

                #use reward estimator to compute estimate
                x = data[['c_time_of_day','c_user']]

                #compute estimate
                estimator.add_example_scikit(x, int(data['a']), list(map(int, list(data['a_vec'])[0])) , float(data['p']), int(data['r']), p_pred, probas, reward_estimator_model)

                estimates.append(estimator.get_estimate())

        idxx += 1

    return estimates

def compute_p_pred(pol_type, data, policy_pi=None):
    """
    Computes the probabilities for predicting actions

    Parameters
    ----------
    pol_type : str
        target policy type used in evaluation
    num_a : int
        number of actions in this setting
    a : int
        the action selected from the logged data

    Returns
    -------
    policy_pi: Policy
        An instance of the Policy class being used
    p_pred: list(float)
        list of predicted action probabilitis

    """
    
    a_const_pred = 2 
    
    if(pol_type=='Constant'):
        policy_pi = policy.Constant(int(data['num_a']), a_const_pred)
        probas, action, p_pred = policy_pi.get_action(0, 0)
        p_pred = 1 if int(data['a']) == action else 0
        
    elif(pol_type=='UniformRandom'):
        policy_pi = policy.UniformRandom(int(data['num_a']))
        probas, action, p_pred = policy_pi.get_action(0, 0)
        p_pred = 1 if int(data['a']) == action else 0
    
    elif(pol_type=='CustomVW'):
        probas, action, p_pred = policy_pi.get_action(data)
        policy_pi.learn_cb_policy(data)
        p_pred = 1 if int(data['a']) == action else 0
        
    return probas, action, p_pred

def compare(policies, estimators, log_data_parsed):
    """
        Given data log collected by using a logging policy, computes the expected reward for a provided set of target policies using a set of estimators provided. Reports the computed values and reports the best policy-estimator combination on that dataset

        Parameters
        ----------
        policies : str or list(str)
            target policies to evaluate
        estimators : str or list(str)
            Estimators to use
        log_data_parsed : file
            Logged data

        Returns
        -------
        None
        """
    best={}


    if not isinstance(policies, list): policies = [policies]
    if not isinstance(estimators, list): estimators = [estimators]

    results = []
    for i, pol in enumerate(policies):
        print("\tPolicy: ", str(policies[i]))
        result = {}
        for j, est in enumerate(estimators):
            if not check_if_dr(est):
                estimates = compute_estimate(pol, est, log_data_parsed)
                print("\t\t"+est+":\t", estimates[-1])
            else:
                estimates = compute_estimate_dr(pol, est, log_data_parsed)
                print("\t\t"+est+":\t", estimates[-1])

            result['policy']=str(pol)
            result['estimator']=est
            result['estimate']=estimates[-1]
            result['estimates']=estimates

            results.append(result.copy())

            if(i==0 and j==0):
                best = result.copy()
            else:
                if(result['estimate']>best['estimate']):
                    best = result.copy()

    print("\tBest result: \n\t\tpolicy: {} \n\t\testimator: {} \n\t\testimate: {}".format(best['policy'], best['estimator'], best['estimate']))

    return best, results


def check_if_dr(est):
    """
        Checks if a policy is doubly-robust based

        Parameters
        ----------
        est : str
            Estimator to use

        Returns
        -------
        bool: True or False
        """
    return est.startswith('dr')

def dr_mode(est):
    """
        {dr_online, dr_batch, dr_seq_online, dr_seq_batch}
        For a doubly-robust based estimator, extracts the type{dr, dr_seq} and mode{online, batch} of the estimator

        Parameters
        ----------
        est : str
            Estimator to use

        Returns
        -------
        str: estimator type{dr, dr_seq}
        str: estimator mode{online, batch}
        """
    dr_list = est.split('_')
    if(len(dr_list)==2):
        return dr_list[0], dr_list[1]
    else:
        return dr_list[0]+"_"+dr_list[1], dr_list[2]

def compute_running_mse(estimates, g_truth_rewards):
    """
        Given logging rewards and estimated rewards, compute windowed RMSE. w=30

        Parameters
        ----------
        estimates: list(float)
        logging_rewards:list(float)

        Returns
        -------
        RMSE:list(float)
        """
    estimates = np.array(estimates)
    g_truth_rewards = np.array(g_truth_rewards)

    MSE=np.zeros(len(g_truth_rewards))
    w=30
    for i in(range(0, len(g_truth_rewards))):
        if(i<=(len(g_truth_rewards)-w)):
            MSE[i]= mean_squared_error(g_truth_rewards[i:i+w], estimates[i:i+w])

    return MSE

def compare_mean_rewards(policy, estimators, log_data_parsed, r_gtruth, N):
    """
        Given data log collected by using a logging policy, computes the expected reward for a provided set of target policies using a set of estimators provided. Reports the computed values and reports the best policy-estimator combination on that dataset

        Parameters
        ----------
        policy : str
            target policy to evaluate
        estimators : str or list(str)
            Estimators to use
        log_data_parsed : file
            Logged data
        r_gtruth: float
            Ground truth r
        Returns
        -------
        best: dict
            best result
        results: list(dict)
            evaluation results
        """
    best={}

    if not isinstance(estimators, list): estimators = [estimators]

    results = []
    print("\tPolicy: ", str(policy))
    
    for j, est in enumerate(estimators):
        result = {}
        g_truth_r = []
        expected_r = []
        
        for n in range(N):
            if not check_if_dr(est):
                estimates = compute_estimate(policy, est, log_data_parsed)
            else:
                estimates = compute_estimate_dr(policy, est, log_data_parsed)

            expected_r.append(estimates[-1])
            g_truth_r.append(r_gtruth)
        
        abs_diff=np.abs(np.mean(expected_r)-r_gtruth)
        print("\t\t"+est+" mean of estimated expected rewards:\t", np.mean(expected_r))
        
        result['policy']=str(policy)
        result['estimator']=est
        result['abs_diff']=abs_diff
        result['mean_expected_r']=np.mean(expected_r)

        results.append(result.copy())

        if(j==0):
            best = result.copy()
        else:
            if(result['abs_diff']<best['abs_diff']):
                best = result.copy()

    print("\tBest result: \n\t\tpolicy: {} \n\t\testimator: {} \n\t\tmean expected reward: {}".format(best['policy'], best['estimator'], best['mean_expected_r']))

    return best, results