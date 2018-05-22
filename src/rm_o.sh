#!/bin/bash
shopt -s globstar
shopt -s nullglob
#for file in **/*.{cu,cuh,cpp,hpp}
#for file in **/*.{cu,cpp}
for file in `find . -name "*.o"`
do
  echo "$file"
rm $file
done
for file in `find . -name "*.so"`
do
  echo "$file"
rm $file
done

rm Makefile
