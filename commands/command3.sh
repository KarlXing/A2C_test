#!/bin/sh

(date; python3.6 -W ignore main.py --env-name "ChopperCommandNoFrameskip-v0"  --num-frames  100000000  --carl-wrapper   --log-evaluation  --lr 1e-4  --complex-model  --reward-mode 0 --activation 0  --track-value-loss) &
(sleep 5; date; python3.6 -W ignore main.py --env-name "ChopperCommandNoFrameskip-v0"  --num-frames  100000000  --carl-wrapper   --log-evaluation  --lr 1e-4  --complex-model  --reward-mode 0 --activation 0  --track-value-loss) &
(sleep 10; date; python3.6 -W ignore main.py --env-name "ChopperCommandNoFrameskip-v0"  --num-frames  100000000  --carl-wrapper   --log-evaluation  --lr 1e-4  --complex-model  --reward-mode 0 --activation 0  --track-value-loss)