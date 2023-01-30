# Better rewards

In this version I tried to tweak the rewards so the agent doesn't try to go
towards the sand. However, the vision of the agent is very limited, only 5
pixels to the front and therefore it almost doesn't see the sand until it's
actually on top of the sand.

Also, bear in mind that the sensors are inputs to the brain, so somehow the
agent already has this information.

# Conclusion
This approach didn't work very well or I wasn't able to guess the right rewards
for the agent.

The way I see it at the moment is that the agent can be considered like an ant
or perhaps a shark, with very little vision but with a great sense of "smell".
Due to this is not able to go around the sand curves and it gets stuck almost
forever.

Perhaps the next approach could be to leave like a trail pheromone so the agent
is able to track where it has been and therefore we don't let the agent to go
back to previous states. However, these pheremones could have like a timeout, so
they dissapear eventually (this might avoid the agent getting trapped in some
cases).