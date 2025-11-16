Lessons learned so far
- GPU not necessarily faster
- vectorizing environment seems to have big effect on training. Doing 10 environments made the max reward be really good but the mean reward is still meh. Doing 1000 environments made the algo not train but I did do way less timesteps
- indexing tensors is 3x faster using python floats instead of pytorch tensors? (e.g. use .item() on the tensor before using as an index)
- stopping episodes early and adding a constant to make the reward fn positive most of the time seems promising