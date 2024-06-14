# Qualitative Analysis
We further improve the qualitative analysis of the results combined with the principles of methods. Specifically, we analyze why our approach can generate some failure scenarios and the baseline method does not. 

A detailed example is shown below to analyze the results:

![图片](./images/analysis.png)

As shown in Figure (a), our method recognizes the current state as the critical state, i.e., the target agent is moving towards the destination, and the NPC is at a suitable distance from the topmost target agent. The NPC agent is perturbed so that it runs perpendicular to the target agent and collides with it.

In the contrast, baselines select a non-critical state for perturbation, as shown in Figure (b). After perturbation, the NPC cannot interfere with the target agent in time, so that the target agent can reach the destination.