#!/bin/sh


(date; python3.6 -W ignore main.py --env-name $1  --num-frames  100000000  --carl-wrapper   --log-evaluation  --lr 1e-4  --complex-model  --reward-mode 0 --activation 0  --track-value-loss) &
(sleep 5; date; python3.6 -W ignore main.py --env-name $2  --num-frames  100000000  --carl-wrapper   --log-evaluation  --lr 1e-4  --complex-model  --reward-mode 0 --activation 0  --track-value-loss) &
(sleep 10; date; python3.6 -W ignore main.py --env-name $3 --num-frames  100000000  --carl-wrapper   --log-evaluation  --lr 1e-4  --complex-model  --reward-mode 0 --activation 0  --track-value-loss)


echo $1 $2 $3 >> command
git rev-parse --short HEAD  >> command
cp command runs/