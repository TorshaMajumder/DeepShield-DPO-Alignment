#!/bin/bash

# Get the absolute path of the current directory
PARENT_DIR=$(pwd)

echo "🚀 Launching DeepShield SageMaker Container ..."

# Run the Docker command
docker run --platform linux/amd64 -it --rm -p 8080:8080 \
    -v "$PARENT_DIR:/opt/ml/model" \
    -e "SAGEMAKER_PROGRAM=code/inference.py" \
    public.ecr.aws/sagemaker/sagemaker-distribution:latest-cpu /bin/bash

# Note: Once inside the container, the user should run:
# cd /opt/ml/model && python mock_sagemaker.py