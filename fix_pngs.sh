
################################################################################

# simple png file fixer script using pngcrush

# (c) 2018 Toby Breckon, Durham University, UK

################################################################################
# check for command line argument

if (test $# -ne 1)
then
  echo "usage: sh ./fix_pngs.sh /path/to/dataset/files"
  exit 1
fi

################################################################################

# set this script to fail on error

set -e

################################################################################
# check for required commands to perform fix

(command -v pngcrush | grep pngcrush > /dev/null) ||
  (echo "Error: pngcrush command not found, cannot fix!";
   echo "install from your package manager or from https://pmt.sourceforge.io/pngcrush/";
   exit 1)

################################################################################
# go the right place to write

cd $1

################################################################################
# perform fix in place

for i in `find * | grep png`; do pngcrush -fix -force $i tmp.png; mv tmp.png $i;done

################################################################################
