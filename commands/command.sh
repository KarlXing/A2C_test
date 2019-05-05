#!/bin/sh

if [ $2 = 1 ]; then
    date; eval $1
elif [ $2 = 2 ]; then
    (date; eval $1) &
    (sleep 5; date; eval $1)
elif [ $2 = 3 ]; then
    (date; eval $1) &
    (sleep 5; date; eval $1) &
    (sleep 10; date; eval $1)
elif [ $2 = 4]; then
    (date; eval $1) &
    (sleep 5; date; eval $1) &
    (sleep 10; date; eval $1) &
    (sleep 15; date; eval $1)
elif [ $2 = 5]; then
    (date; eval $1) &
    (sleep 5; date; eval $1) &
    (sleep 10; date; eval $1) &
    (sleep 15; date; eval $1) &
    (sleep 20; date; eval $1)
else
    echo "Invalid Loop"
fi

echo $1 >> command
git rev-parse --short HEAD  >> command
cp command runs/