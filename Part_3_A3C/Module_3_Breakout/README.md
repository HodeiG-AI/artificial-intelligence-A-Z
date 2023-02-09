This project is originally based on https://github.com/ikostrikov/pytorch-a3c

He highly recommends to check a sychronous version and other algorithms: [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr).

In his experience, A2C works better than A3C and ACKTR is better than both of them. Moreover, PPO is a great algorithm for continuous control. Thus, I recommend to try A2C/PPO/ACKTR first and use A3C only if you need it specifically for some reasons.

Also read [OpenAI blog](https://blog.openai.com/baselines-acktr-a2c/) for more information.


The actual tutorial code was copied from https://github.com/Mattemyo/artificial-intelligence-az