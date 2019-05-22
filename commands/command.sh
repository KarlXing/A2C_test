#!/bin/sh

command1="python3.6 -W ignore main.py --env-name $1 --num-frames  $2  --log-evaluation  --lr 1e-4  --reward-mode 0" 
command2="python3.6 -W ignore main.py --env-name $1 --num-frames  $2  --log-evaluation  --lr 1e-4  --reward-mode 1" 
command3="python3.6 -W ignore main.py --env-name $1 --num-frames  $2  --log-evaluation  --lr 1e-4  --reward-mode 2" 

(date; eval $command1) &
(sleep 5; date; eval $command1) &
(sleep 10; date; eval $command1) &
(sleep 15; date; eval $command2) &
(sleep 20; date; eval $command2) &
(sleep 25; date; eval $command2) &
(sleep 30; date; eval $command3) &
(sleep 35; date; eval $command3) &
(sleep 40; date; eval $command3)

echo $command1 >> command
echo $command2 >> command
echo $command3 >> command
git rev-parse --short HEAD  >> command
cp command runs/