# Download DIV2K dataset
mkdir dataset

wget data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip

unzip DIV2K_train_HR.zip -d ./dataset
unzip DIV2K_valid_HR.zip -d ./dataset

rm DIV2K_train_HR.zip
rm DIV2K_valid_HR.zip