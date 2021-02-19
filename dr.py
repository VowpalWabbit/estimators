import os
import ds_parse
from vowpalwabbit import pyvw


class Estimator:
    """
    A class for the Doubly Robust Estimator


    Attributes
    ----------
    data: dict
        dictionary containing various attributes
    data['v_hat']: float
        Doubly Robust Estimator estimate
    data['N']: int
        total number of samples in bin

    Methods
    -------
    add_example(self, x , a, a_vec, p_log, r, p_pred, policy_probas, reward_estimator_model, count=1):
        Implements the doubly robust estimator equation and computes the summation in the equation
    def get_estimate(self):
        Computes the mean of the estimated value from the doubly robust estimator equation
    reward_estimator(self, train_data, reward_estimator_model, mode='batch', load_model=False):
        Trains a learner to use in the estimation of a reward
    """
    def __init__(self):
        """
        Constructs attributes of the Doubly Robust Estimator

        Parameters
        ----------
            data: dict
                dictionary containing various attributes
            data['v_hat']: float
                Doubly Robust Estimator estimate
            data['N']: int
                total number of samples in bin
        """
        
        self.data = {'v_hat':0.,'N':0}

    def add_example(self, x , a, a_vec, p_log, r, p_pred, policy_probas, reward_estimator_model, count=1):
        """
        Implements the doubly robust estimator equation and computes the summation in the equation

        Parameters
        ----------
        x : str
            X example from logging policy in vw format to pass to the reward estimator model to predict a reward
        a : int
            Action from logging policy
        a_vec : list(int)
            List of actions in logging policy
        p_log : float
            Probability of choosing an action in the given a context. From logging policy
        r : int
            Reward from logging policy
        p_pred : float
            More info to be displayed (default is None)
        policy_probas : list(float)
            List of probabilities of choosing the actions in the target policy being evaluated
        reward_estimator_model : vw model (pyvw.vw)
            A vw model that estimates a reward for a given context and reward, (x, a) --> r
            
        Returns
        -------
        None
        """
        self.data['N'] += count

        r_hat = sum(policy_probas[i] * reward_estimator_model.predict("|ANamespace Action:" + str(ac)+ x ) for i, ac in enumerate(a_vec))
        self.data['v_hat'] += r_hat

        p_over_p = p_pred/p_log
        value_bias_correction = r - reward_estimator_model.predict("|ANamespace Action:" + str(a) + x )

        self.data['v_hat'] += (p_over_p * value_bias_correction)

    def get_estimate(self):
        """
        Trains a learner to use in the estimation of a reward

        Parameters
        ----------
        None
            
        Returns
        -------
        float: policy value
        """
        v_hat = 0
        if(self.data['N'])>0:
            v_hat = self.data['v_hat'] / self.data['N']
        return v_hat

    def reward_estimator(self, train_data, reward_estimator_model, mode='batch', load_model=False):
        """
        Computes the mean of the estimated value from the doubly robust estimator equation

        Parameters
        ----------
        train_data : str
            (x, a, r) data from the logging policy to train the model
            if mode=batch, train_data is a list of training examples
            if mode=online, train_data is one training example
                    
        reward_estimator_model : vw model (pyvw.vw)
            A vw model that is trained from the training example(s)
        mode : float
            The mode of the doubly robust estimator, can be {batch, online}
        load_model : bool
            indicator variable, if true then do not train model afresh, otherwise train afresh
            
        Returns
        -------
        vowpal wabbit model
        """
        if(mode=='batch'):
            if (not load_model):
                for example in train_data:
                    reward_estimator_model.learn(example)
                    
                reward_estimator_model.save("../models/reward_estimator_batch.model")

            else:
                reward_estimator_model = pyvw.vw("-i ../models/reward_estimator_batch.model")# --quiet

        elif(mode=='online'):
            if (not load_model):
                reward_estimator_model.learn(train_data)

            else:
                pass

        return reward_estimator_model

    def reward_estimator_scikit(self, train_data, clf, classes, mode='batch', load_model=False):
        """
        Computes the mean of the estimated value from the doubly robust estimator equation

        Parameters
        ----------
        train_data : str
            (x, a, r) data from the logging policy to train the model
            if mode=batch, train_data is a list of training examples
            if mode=online, train_data is one training example
                    
        reward_estimator_model : vw model (pyvw.vw)
            A vw model that is trained from the training example(s)
        mode : float
            The mode of the doubly robust estimator, can be {batch, online}
        load_model : bool
            indicator variable, if true then do not train model afresh, otherwise train afresh
            
        Returns
        -------
        vowpal wabbit model
        """
        
        X = train_data[['a','c_time_of_day','c_user']]
        Y = train_data[['r']]
        X = X.astype({"a": int, "c_time_of_day": int, "c_user": int})
        Y = Y.astype({"r": int})
                
        if(mode=='batch'):
            if (not load_model):
                reward_estimator_model = clf.fit(X, Y.values.ravel())
                
            else:
                pass
            
        elif(mode=='online'):
            if (not load_model):
                reward_estimator_model = clf.partial_fit(X, Y.values.ravel(), classes)
            else:
                pass

        return reward_estimator_model
    
    def add_example_scikit(self, x , a, a_vec, p_log, r, p_pred, policy_probas, reward_estimator_model, count=1):
            """
            Implements the doubly robust estimator equation and computes the summation in the equation

            Parameters
            ----------
            x : str
                X example from logging policy in vw format to pass to the reward estimator model to predict a reward
            a : int
                Action from logging policy
            a_vec : list(int)
                List of actions in logging policy
            p_log : float
                Probability of choosing an action in the given a context. From logging policy
            r : int
                Reward from logging policy
            p_pred : float
                More info to be displayed (default is None)
            policy_probas : list(float)
                List of probabilities of choosing the actions in the target policy being evaluated
            reward_estimator_model : vw model (pyvw.vw)
                A vw model that estimates a reward for a given context and reward, (x, a) --> r

            Returns
            -------
            None
            """
            self.data['N'] += count
            
            x = x.reset_index(drop=True)
            r_hat=0
            
            for i, ac in enumerate(a_vec):
                x['a']=list([a_vec[i]])
                r_hat = policy_probas[i] * reward_estimator_model.predict(x)
                self.data['v_hat'] += r_hat

            p_over_p = p_pred/p_log
            
            x['a']=list([a])
            value_bias_correction = r - reward_estimator_model.predict(x)

            self.data['v_hat'] += (p_over_p * value_bias_correction)