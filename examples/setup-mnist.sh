#!/bin/bash

# Define the target directory and base URL
TARGET_DIR="data"
BASE_URL="https://ossci-datasets.s3.amazonaws.com/mnist"

# Create the directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# List of files to download
FILES=(
  "train-images-idx3-ubyte"
  "train-labels-idx1-ubyte"
  "t10k-images-idx3-ubyte"
  "t10k-labels-idx1-ubyte"
)

echo "Starting MNIST dataset setup..."

for FILE in "${FILES[@]}"; do
  # Check if the uncompressed file already exists
  if [ -f "$TARGET_DIR/$FILE" ]; then
    echo "Skipping $FILE (already exists)"
  else
    echo "Downloading and decompressing $FILE..."
    # Download via curl and pipe directly into gunzip to save to the target directory
    curl -L "$BASE_URL/$FILE.gz" | gunzip > "$TARGET_DIR/$FILE"
    
    if [ $? -eq 0 ]; then
      echo "Successfully saved to $TARGET_DIR/$FILE"
    else
      echo "Error: Failed to process $FILE"
    fi
  fi
done

echo "MNIST setup complete."