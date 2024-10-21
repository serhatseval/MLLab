# import required module
import os
import re
from plot_spectogram import main
# assign directory
directory = "C:\\Users\\Marcel\\Desktop\\IML\\daps\\daps\\ClipsForCNN"
x = 0
# iterate over files in
# that directory
for root, dirs, files in os.walk(directory):
    for filename in files:
        if re.match(r'^(f1_|f7|m3|m6|m8)', filename):
            #main(os.path.join(root, filename))
            x=x+1
            print(filename)