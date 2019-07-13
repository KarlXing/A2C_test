#!/bin/sh

command1="python3.6 -W ignore main.py --env-name $1 --num-frames  $2  --log-evaluation  --value-loss-coef 1.0 ---track-value-loss  --track-grad " 
command2="python3.6 -W ignore main.py --env-name $1 --num-frames  $2  --log-evaluation  --value-loss-coef 0.5  --track-value-loss  --track-grad" 
command3="python3.6 -W ignore main.py --env-name $1 --num-frames  $2  --log-evaluation  --value-loss-coef 0.2  --track-value-loss  --track-grad" 

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