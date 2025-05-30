#! /bin/bash

# Original download path (invalid)
# wget https://vision.eecs.yorku.ca/WebShare/COIN_s3d.zip
# unzip COIN_s3d.zip

# The above download path is unvalid, so we use the following instead:
# https://drive.google.com/file/d/1Bq0iJoLKASOgXeNRmoPSJtzQy01IGKqh/view?usp=sharing

# Install gdown if not installed
pip install gdown --quiet

# Download from Google Drive
echo "Downloading COIN dataset from Google Drive..."
gdown https://drive.google.com/uc?id=1Bq0iJoLKASOgXeNRmoPSJtzQy01IGKqh

# Extract the downloaded file
echo "Extracting files..."
unzip -q COIN_s3d.zip

echo "Download and extraction completed!"