#! /bin/bash

# Original download path (invalid)
# wget https://vision.eecs.yorku.ca/WebShare/NIV_s3d.zip
# unzip NIV_s3d.zip

# The above download path is unvalid, so we use the following instead:
# https://drive.google.com/file/d/1nmsYW0hRGWChCfuTnsvn8lfW2YWmFTfA/view?usp=sharing

# Install gdown if not installed
pip install gdown --quiet

# Download from Google Drive
echo "Downloading NIV dataset from Google Drive..."
gdown https://drive.google.com/uc?id=1nmsYW0hRGWChCfuTnsvn8lfW2YWmFTfA

# Extract the downloaded file
echo "Extracting files..."
unzip -q NIV_s3d.zip

echo "Download and extraction completed!"