#!/bin/sh

command1="python3.6 -W ignore main.py --env-name $1 --num-frames  $2  --log-evaluation  --lr 1e-4  --reward-mode 0  --track-value-loss" 
command2="python3.6 -W ignore main.py --env-name $1 --num-frames  $2  --log-evaluation  --lr 1e-4  --reward-mode 1  --track-value-loss" 
command3="python3.6 -W ignore main.py --env-name $1 --num-frames  $2  --log-evaluation  --lr 1e-4  --reward-mode 2  --track-value-loss" 

(date; eval $command1; echo "done") &
(sleep 5; date; eval $command1; echo "done") &
(sleep 10; date; eval $command1; echo "done") &
(sleep 15; date; eval $command2; echo "done") &
(sleep 20; date; eval $command2; echo "done") &
(sleep 25; date; eval $command2; echo "done") &
(sleep 30; date; eval $command3; echo "done") &
(sleep 35; date; eval $command3; echo "done") &
(sleep 40; date; eval $command3; echo "done")

wait 

echo $command1 >> command
echo $command2 >> command
echo $command3 >> command
git rev-parse --short HEAD  >> command
cp command runs/