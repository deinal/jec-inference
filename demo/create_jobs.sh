#!/bin/bash

mkdir -p ./jobs

for f in {001..005}
do
  cat job_template.yaml | sed "s/\$ROOTFILE/$f.root/" > ./jobs/job_$f.yaml
done

kubectl create -f ./jobs