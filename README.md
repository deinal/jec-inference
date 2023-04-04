# jec-inference

Results for Jet Energy Calibration with Deep Learning as a Kubeflow Pipeline

## Notebook setup

Start a new notebook server on https://ml-staging.cern.ch/_/jupyter/.

Install requirements `pip install -r requirements.txt`.

There are notebooks with data vizualisation, hyperparameter correlation, and inference results.

## Meetup demo

Parallel inference demo for an MLOps meetup https://youtu.be/AWZT9ZYgohY

Enter `./demo`

Build inference image
```
docker build . -t registry.cern.ch/ml/jec-inference
docker push registry.cern.ch/ml/jec-inference
```
Serve a PFN model stored on s3
  - `kubectl apply -f serve.yaml`

Parallel inference on Kubeflow using Kuberenetes jobs
  - https://kubernetes.io/docs/tasks/job/

Tried indexed queue with `indexed_job.yaml`. However, it used file from index zero for all jobs...

However, expansion of jobs worked
  - https://kubernetes.io/docs/tasks/job/parallel-processing-expansion/
  - Run: `./create_jobs.sh`
  - Monitor: `watch -n 1 kubectl get pods`
  - Cleanup: `kubectl delete job -l jobgroup=inference`
