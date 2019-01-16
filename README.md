# Estimators Library

In contextual bandits, a learning algorithm repeatedly observes a context, takes an action, and observes a reward for the chosen action. An example is content personalization: the context describes a user, actions are candidate stories, and the reward measures how much the user liked the recommended story. In essence, the algorithm is a policy that picks the best action given a context.

Given different policies, the metric of interest is their reward. One way to measure the reward is to deploy such policy online and let it choose actions (for example, recommend stories to users). However, such online evaluation can be costly for two reasons: It exposes users to an untested, experimental policy; and it doesn't scale to evaluating multiple target policies.

The alternative is off-policy evaluation: Given data logs collected by using a logging policy, off-policy evaluation can estimate the expected rewards for different target policies and provide confidence intervals around such estimates.

This repo collects estimators to perform such off-policy evaluation.
