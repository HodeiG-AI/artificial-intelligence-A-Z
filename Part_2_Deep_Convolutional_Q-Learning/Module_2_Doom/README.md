# How to render gym games

env.render() should be enough to render the game. However, the window that will
be created might not respond. In order to fix that we need get the event from
pygame:

```python
import pygame

env.render()
pygame.event.get()  # This will make the window responsive again
```

See: https://stackoverflow.com/a/20166290/3585964