# 1. Build the source distribution (creates a dist/ folder)
python setup.py sdist --formats=gztar

# 2. Upload the created tarball to your GCS bucket
gcloud storage cp dist/trainer-0.1.tar.gz gs://neuroagro1-gcp-mlops/