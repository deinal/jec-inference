#!/bin/bash

mkdir -p ./jobs

for f in 120.root 121.root 122.root 124.root 125.root 126.root 127.root 128.root 129.root
do
  cat job_template.yaml | sed "s/\$ROOTFILE/$f/" > ./jobs/job_$f.yaml
done

kubectl create -f ./jobs