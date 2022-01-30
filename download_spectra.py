import os

os.system("pushd dr12q/spectra")
os.system("rsync --info=progress2 -h --no-motd --files-from=file_list rsync://data.sdss.org/dr12/boss/spectro/redux/ . 2> /dev/null")
os.system("popd")