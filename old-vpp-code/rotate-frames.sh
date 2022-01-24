#!/bin/bash

process_dir()
{
dirname=$1

for file in ${dirname}/*.jpg
do
  convert ${file} -rotate 90 ${file}.rot
done

for file in ${dirname}/*.jpg
do
  mv ${file}.rot ${file}
done
}

process_dir "test_washing/0"
process_dir "test_washing/1"
process_dir "trainval_washing/0"
process_dir "trainval_washing/1"
