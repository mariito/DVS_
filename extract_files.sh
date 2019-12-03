#!/bin/bash

cd /media/maria/0C7ED7537ED733E4/Downloads-/temp/tar
mkdir all

for f in *.tar.gz; do tar xf "$f" -C all; done


mkdir train test

entries=($(shuf -i 1-421 -n 42))
for i in "${entries[@]}"
do
	echo "$i"
	printf -v j "%05d" $i
	mv all/$j test
	
done

mv all/* train


for d in train/* ; do
	echo $d
	number=$(echo "$d" | sed 's/[^0-9]*//g')
	echo $number
	mkdir train/$number train/$number/GT train/$number/unstable


	ffmpeg -i train/$number/GT.mp4 -q:v 2 train/$number/GT/%05d.jpg
	ffmpeg -i train/$number/unstable.mp4 -q:v 2 train/$number/unstable/%05d.jpg
done


