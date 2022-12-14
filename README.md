# jec-inference

## Notebook setup

Start notebook on https://ml-staging.cern.ch/_/jupyter/ with the `jupyter-scipy` image.

Install requirements `pip install -r requirements.txt`.

Now you can run the `inference.ipynb` notebook.

## Jobs

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

Instead tried expansion of jobs 
  - https://kubernetes.io/docs/tasks/job/parallel-processing-expansion/
  - Run: `./create_jobs.sh`
  - Monitor: `watch -n 1 kubectl get pods`
  - Cleanup: `kubectl delete job -l jobgroup=inference`
