#!/bin/bash
# Setup DVC for model and data versioning

set -euo pipefail

echo "Setting up DVC..."

# Initialize DVC (if not already initialized)
if [ ! -d ".dvc" ]; then
    echo "Initializing DVC..."
    dvc init
else
    echo "DVC already initialized"
fi

# Configure remote storage
read -p "Select remote storage (s3/gs/azure): " STORAGE_TYPE

case $STORAGE_TYPE in
    s3)
        read -p "Enter S3 bucket name: " BUCKET_NAME
        read -p "Enter AWS region [us-west-2]: " AWS_REGION
        AWS_REGION=${AWS_REGION:-us-west-2}

        dvc remote add -d myremote s3://$BUCKET_NAME/dvc-storage
        dvc remote modify myremote region $AWS_REGION
        echo "✓ S3 remote configured: s3://$BUCKET_NAME/dvc-storage"
        ;;

    gs)
        read -p "Enter GCS bucket name: " BUCKET_NAME

        dvc remote add -d myremote gs://$BUCKET_NAME/dvc-storage
        echo "✓ GCS remote configured: gs://$BUCKET_NAME/dvc-storage"
        ;;

    azure)
        read -p "Enter Azure container name: " CONTAINER_NAME
        read -p "Enter Storage account: " ACCOUNT_NAME

        dvc remote add -d myremote azure://$CONTAINER_NAME/dvc-storage
        dvc remote modify myremote account_name $ACCOUNT_NAME
        echo "✓ Azure remote configured"
        ;;

    *)
        echo "Invalid storage type"
        exit 1
        ;;
esac

echo ""
echo "DVC setup complete!"
echo ""
echo "Next steps:"
echo "  1. Add files to DVC: dvc add data/dataset.csv"
echo "  2. Commit .dvc files: git add data/dataset.csv.dvc .gitignore && git commit"
echo "  3. Push to remote: dvc push"
