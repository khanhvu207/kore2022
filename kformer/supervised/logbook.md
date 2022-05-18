# Logbook

## May 18, 2022
The code is running and the model does not learn at all!  
I know this is the case by observing the losses and accuracies:  
- Train loss does not decrease.  
- Action accuracy is exactly 0.666 which is the majority class :D
- Token accuracy is around 0.066.  

**Hypothesis & possible solutions**
- The input does not contain meaningful information for the task.
  - The information about the current shipyard is burried in the feature map. So perhaps we can center the feature map to the shipyard location
  - See this: https://stats.stackexchange.com/questions/352036/what-should-i-do-when-my-neural-network-doesnt-learn
  - Try to see if the neural net can overfit 5-10 batches. (http://karpathy.github.io/2019/04/25/recipe/)
  - Represent shipyards as tokens.
- The code is not bug-free.
  - Perform deep proofreading.