import numpy.random
from vowpalwabbit import pyvw
import numpy as np

class Policy:
    r"""
    An abstract class representing a Reinforcement Learning (RL) policy,
    mapping a state to a probability for each possible action.  A policy can be
    stateless or stateful.
    All other policies should subclass it. All subclasses should
    override ``get_action_probas``, that returns the probilities associated
    with each possible action.
    Non stateless policies should cache their results in
    Observation.target_probas.
    """

    def __init__(self, stateless=True):
        r"""
        Arguments:
            stateless: is the policy stateless? When it's not, some estimators
                may run it to cache the results in an Observation
        """
        self.stateless = stateless

    def get_action_probas(self, index, observation):
        r"""
        Returns the probabilities associated with each possible action.
        Arguments:
            index: index in the trace (in 1 ... H)
            observation: an Observation
        Returns:
            A list of probabilities (ps where len(ps) = action number,
            sum(ps) == 1, and p >=0 for p in ps)
        """
        raise NotImplementedError

    def get_action(self, index, observation):
        r"""
        Returns a chosen action and its associated probility.
        Arguments:
            index: index in the trace (in 1 ... H).
            observation: an Observation.
        Returns:
            (a, p): a tuple of the chosen action in (1 ... action_num) and its
            probability.
        """
        p = self.get_action_probas(index, observation)
        a = numpy.random.choice(len(p), p=p)
        return (p, a+1, p[a])

class Constant(Policy):
    r"""
    A constant policy that always returns the same action with probability 1.
    """
    def __init__(self, action_num, action):
        r"""
        Arguments:
            action_num: the total number of actions
            action: the action returned with probability 1, indexed in
                1 ... action_num
        """
        super().__init__(stateless=True)

        self.action_num = action_num
        self.action = action

        self.ps = [0] * action_num
        self.ps[action-1] = 1.0

    def get_action_probas(self, index, observation):
        return self.ps

class UniformRandom(Policy):
    r"""
    A policy that returns a subset of the actions uniformly at random.
    """
    def __init__(self, action_num, possible_actions=None):
        r"""
        Arguments:
            action_num: the total number of actions
            action: the actions returned uniformly at random, indexed in
                1 ...  action_num
        """
        super().__init__(stateless=True)
        self.action_num = action_num

        if possible_actions == None:
            possible_actions = [a+1 for a in range(action_num)]
        self.possible_actions = possible_actions

        p = 1. / len(self.possible_actions)
        self.ps = [0] * action_num
        for a in self.possible_actions:
            self.ps[a-1] = p

    def get_action_probas(self, index, observation):
        return self.ps

class RoundRobin(Policy):
    r"""
    A policy that choses each action in turn with prba 1.
    This is an example of a stateful policy.
    """
    def __init__(self, action_num):
        r"""
        Arguments:
            action_num: the total number of actions
        """
        super().__init__(stateless=False)
        self.action_num = action_num
        self.next_action = 0

    def get_action_probas(self, index, observation):
        r"""
        Example of stateful policies, get_action_probas should cache the
        results in the observation's target_probas field.
        """
        if observation.target_probas is not None:
            # The probaes have been cached, return them.
            return observation.target_probas

        # Compute probas
        p = [0.] * self.action_num
        p[self.next_action] = 1.0
        # Cache result
        observation.target_probas = p

        # Update state
        self.next_action = (self.next_action + 1) % self.action_num

        return p

class ReturnTargetProbas(Policy):
    r"""
    A policy that just returns observation.target_probas.
    It assumnes they exist in Observation.
    """
    def __init__(self):
        super().__init__(stateless=True)

    def get_action_probas(self, index, observation):
        return observation.target_probas

#class CustomPolicy(Policy):
#    pass
    
class VWPolicy(Policy):
    r"""
    A policy that instantiates a VW policy and returns the action probabilities.
    """
    def __init__(self, vw_model):
        super().__init__(stateless=True)
        self.vw_model=vw_model 
        
    def learn_cb_policy(self, data):
        self.vw_model.learn(str(data['a'])+":"+str(data['cost'])+":"+str(data['p'])+" | "+data['XNamespace'])

    def get_action_probas(self, data):
        target_probas=self.vw_model.predict("| "+ data['XNamespace'])
        return target_probas
                                            
    def get_action(self, data):
        r"""
        Returns a chosen action and its associated probility.
        Arguments:
            data: 
            
        Returns:
            (p, a, p[a]): a tuple of the chosen action in (1 ... action_num) and its
            probability.
        """
        p = self.get_action_probas(data)
        #normalizing p to ensure it always sums to 1
        c=np.array(p)
        p=c/sum(p)
                
        a = np.random.choice(len(p), p=p)
        return (p, a+1, p[a])                                        
