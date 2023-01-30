# Retrospective Experience Replay
pd: I called this Retrospective Experience Replay (synonim of Hindsight) as we
update the Experience Replay rewards once we know the final outcome.

The idea is to update the rewards only when we know that the car reached the
destination.

I have been playing with the idea that we shouldn't be setting the rewards
randomly, as we don't know how can this affect to the learning process of the
agent. Ideally, we would like to give just a unique reward to the agent (+1)
when it reaches the destination.

In order to do this, we would have a second experience replay component that
would keep the samples that haven't been rewarded yet and once we reach the
destination we could reward all the experiences with a value (+1).

To make the trip even more efficient, we could calculate the distance taken to
reach the destination relative to the starting point and calculate the final
reward using that score. So if for instance the car starts in the bottom right
corner and the distance to the top left corner is 30 but the car needed 100
moves, the score could be 30/100=0.3. However if the car needed 50 moves the
score could be calculated as 30/50=0.3.

The reason to calculate the relative distance is that during the 1st epoch, the
car might get initialised in the centre and therefore it will need less moves
to reach the destination.

I have added the link of the Alpha Zero algoright below, as adding the rewards
after the game has finished, looks like something that has already been done
before.

https://web.stanford.edu/~surag/posts/alphazero.html

# Hindsight Experience Replay
https://github.com/orrivlin/Hindsight-Experience-Replay---Bit-Flipping
https://github.com/orrivlin/Navigation-HER

# Prioritized Experience Replay
As per this blog, it doesn't look like it works very well.

https://towardsdatascience.com/how-to-implement-prioritized-experience-replay-for-a-deep-q-network-a710beecd77b
https://github.com/Guillaume-Cr/lunar_lander_per


# Conclusions

The idea of just updating the rewards after reaching the destination didn't
solve any issues.

The car still gets stuck in some curves and as it doesn't have the full
knowledge of the map, it cannot figure out what to do when it gets stuck in a
curver.

In another experiment, I will change how the rewards are given to the car, as
I believe I found some contradictions. On the hand the car will get possitive
reward when getting closer to the destination. But on the other hand it doesn't
get negative reward when it detects sand in front. If the car knows that there
is sand in front, why should it receive a positive reward? After all it will get
stuck in the sand.