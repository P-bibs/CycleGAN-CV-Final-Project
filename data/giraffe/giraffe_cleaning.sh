#!/bin/bash
file="output2.txt"
while IFS= read -r line
do
    # display $line or do somthing with $line
    find . -name "$line" -delete
done <"$file"