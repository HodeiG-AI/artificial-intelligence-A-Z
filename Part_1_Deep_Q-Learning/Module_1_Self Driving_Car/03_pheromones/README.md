# Pheremones reward

In this version, I have added something similar to the pheromones used by the
ants to find the way to return.

After the implementation the agent works very well as it gets penalised when going
towards the pheromone trace left behind. Perhaps, this makes me think that this
is something different to what pheromones are used for, but it adds a small
memory to the agent to remember where is coming from, so it knows it's not worth
going back to explore at least the path that has the pheromones.