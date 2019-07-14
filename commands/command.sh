#!/bin/sh

command1="python3.6 -W ignore main.py --env-name $2 --num-frames  $1  --log-evaluation   --track-value-loss  --track-grad" 
command2="python3.6 -W ignore main.py --env-name $3 --num-frames  $1  --log-evaluation   --track-value-loss  --track-grad" 
command3="python3.6 -W ignore main.py --env-name $4 --num-frames  $1  --log-evaluation   --track-value-loss  --track-grad" 
command4="python3.6 -W ignore main.py --env-name $5 --num-frames  $1  --log-evaluation   --track-value-loss  --track-grad" 

(date; eval $command1; echo "done") &
(sleep 5; date; eval $command2; echo "done") &
(sleep 10; date; eval $command3; echo "done") &
(sleep 15; date; eval $command4; echo "done")


wait 
echo "mean"
echo $command1 >> command
echo $command2 >> command
echo $command3 >> command
echo $command4 >> command

git rev-parse --short HEAD  >> command
cp command runs/