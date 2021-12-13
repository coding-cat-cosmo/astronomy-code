#import numpy as np
#in_arr = [2, 0, 25]
#in_arr = np.invert(in_arr)
#print(in_arr)
#outf.close()
import os
#from pathlib import Path
#p = Path(os.getcwd())
#gets parent path
#base_directory = p.parent
# Parent Directory path
#parent_dir = str(p.parent)
#place = os.path.join(parent_dir, "")
os.system("pushd dr12q/spectra")
os.system("rsync --info=progress2 -h --no-motd --files-from=file_list rsync://data.sdss.org/dr12/boss/spectro/redux/ . 2> /dev/null")
os.system("popd")
#print(os.system("rsync --info=progress2 -h --no-motd --files-from=file_list rsync://data.sdss.org/dr12/boss/spectro/redux/ . 2> /dev/null"))

