#!/bin/sh

(date; eval $1) &
(sleep 5; date; eval $1)

echo $1 >> command
cp command runs/
