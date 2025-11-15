Lessons learned so far
- GPU not necessarily faster
- vectorizing environment seems to have big effect on training. Doing 10 environments made the max reward be really good but the mean reward is still meh. Doing 1000 environments made the algo not train but I did do way less timesteps