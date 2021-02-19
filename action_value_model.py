class ActionValueModel:
    r"""
    An abstract class representing an action value model for a fixed policy
    \pi, a.k.a. Q_\pi, in a Reinforcement Learning (RL) setting.

    All other action value models should subclass it. All subclasses should
    override ``q``, that returns the Q value prediction for an Observation
    and an action Q(obs.state, action) ; and ``update'', which updates the
    Q value model given two subsequent observations.
    """

    def __init__(self, horizon, action_num, policy):
        r"""
        Initializes variables to store episode data and all the Q models.

        Arguments:
            horizon: RL horizon (fixed horizon setting), a.k.a H.
            actions_num: number of possible actions (indexing starts at 1)
            policy: a Policy object for which we want to learn Q.
        """
        self.horizon = horizon
        self.action_num = action_num
        self.policy = policy

    def q(self, index, observation, action):
        r"""
        Tge prediction for Q_\pi at index, for the state action pair
        (observation, action).

        Arguments:
            index: the number of steps already passed in the episode,
                   in 1 ... H
            obseravtion: an Observation, where obs.state is s_i in Q_\pi(s_i, a_i)
            action: the action a in Q_\pi(s_i, a_i)

        Returns the predicted cumulative reward for the last t observations of
        the episode, playing a then \pi
        """
        raise NotImplementedError

    def v(self, index, observation):
        r"""
        Integrates Q_\pi over all action and proabs for observation at index
        to return the cumulative reward to the end of the episode, playing \pi.

        Arguments:
            index: the number of steps already passed in the episode,
                   in 1 ... H
            obseravtion: an Observation, where obs.state is s_i in V_\pi(s_i)

        Returns the predicted cumulative reward for the last t observations of
        the episode
        """
        if index > self.horizon:
            return 0

        v = 0
        for action, p in enumerate(self.policy.get_action_probas(index, observation)):
            v += p * self.q(index, observation, action+1)
        return v

    def update(self, index, observation, next_observation):
        r"""
        Updates Q_\pi at index, with a new observation and its next state.

        Arguments:
            index: the number of steps already passed in the episode,
                   in 1 ... H
            obseravtion: an Observation of state, action, proba, reward
            next_observation: an Observation used to get the next state after
                observation in Q learning
        """
        raise NotImplementedError
