#!/bin/bash

data="$(ls -1 "$1"/*.data)"

for i in $data; do
    echo "$i"
    ../convert/convert convert image -i "$i" -o "$2/$(basename "$i").png" --mode decode
done

