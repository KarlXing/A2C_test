#!/bin/sh

(date; $1) &
(sleep 5; date; $1) &
(sleep 10; date; $1) &
(echo $1 >> command) 
cp command runs/