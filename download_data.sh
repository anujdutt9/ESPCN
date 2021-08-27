# Download DIV2K Training and Validation datasets
mkdir dataset

echo "Downloading training and validation zip files."
wget data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip

echo "Extracting training and validation zip files."
unzip DIV2K_train_HR.zip -d ./dataset
unzip DIV2K_valid_HR.zip -d ./dataset

echo "Cleanup."
rm DIV2K_train_HR.zip
rm DIV2K_valid_HR.zip

echo "Done."