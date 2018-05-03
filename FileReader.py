import io
import os
from scipy import misc
#Todo: Implement A filereader to convert from image to array in n*m format

def get_all_photos():
    rootdir = "chars74k-lite"
    dirs1 = None
    files1 = []
    images = []
    for subdirs, dirs, files in os.walk(rootdir):
        if(len(dirs) > 0):
            dirs1 = dirs
        files1.append(files)

    for directory in range(len(dirs1)):
        images.append([])
        dir_name = dirs1[directory]
        for file_name in os.listdir(rootdir + "/" + dir_name):
            file_dir = rootdir + "/" + dir_name + "/" + file_name
            jpg = misc.imread(file_dir)
            images[directory].append(jpg)

    return images
