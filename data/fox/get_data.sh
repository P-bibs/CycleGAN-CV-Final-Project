#!/bin/bash
file="fox_images"
while IFS= read -r line
do
    # display $line or do somthing with $line
    wget $line
done <"$file"