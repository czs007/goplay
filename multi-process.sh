#!/bin/bash

for ((i=0; i<144; i++))
do
	echo $i
	./test > "cost-$i.txt" &
done
wait
