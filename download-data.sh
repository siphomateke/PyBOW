################################################################################

# simple data downloader / unpacker - (c) 2018 Toby Breckon, Durham University, UK

################################################################################

# set this script to fail on error

set -e

# check for required commands to download and md5 check

(command -v curl | grep curl > /dev/null) ||
  (echo "Error: curl command not found, cannot download!")

(command -v md5sum | grep md5sum > /dev/null) ||
  (echo "Error: md5sum command not found, md5sum check will fail!")

################################################################################

STARTING_DIR=`pwd`

################################################################################

## INRIA Pedestrian Dataset

################################################################################

URL_PERSON=ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar

DIR_LOCAL_TARGET_PERSON=/tmp/pedestrian

PERSON_FILE_NAME=INRIAPerson.tar

DIR_NAME_UNPACKED=INRIAPerson
PERSON_MD5_SUM=6af009c6386c86f78f77e81003df84dc

################################################################################

# perform download

echo "Downloading pedestrian data set models..."

mkdir -p $DIR_LOCAL_TARGET_PERSON

TARGET=$DIR_LOCAL_TARGET_PERSON/$PERSON_FILE_NAME

curl --progress-bar $URL_PERSON > $TARGET

################################################################################

# perform md5 check and move to required local target directory

cd $DIR_LOCAL_TARGET_PERSON

echo "checking the MD5 checksum for downloaded data ..."

CHECK_SUM_CHECKPOINTS="$PERSON_MD5_SUM  $PERSON_FILE_NAME"

echo $CHECK_SUM_CHECKPOINTS | md5sum -c

echo "Unpacking the tar file..."

tar -xvf $PERSON_FILE_NAME

chmod -R +w $DIR_LOCAL_TARGET_PERSON

echo "Tidying up..."

ln -s $DIR_LOCAL_TARGET_PERSON $STARTING_DIR/pedestrian

# mv $DIR_NAME_UNPACKED/* .

rm $TARGET # && rm -r $DIR_NAME_UNPACKED

################################################################################

echo "... completed -> required pedestrian data in $DIR_LOCAL_TARGET_PERSON/"

################################################################################

# reset

cd $STARTING_DIRPERSON

################################################################################
