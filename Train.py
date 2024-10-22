# import required module
import os
import re
from plot_spectogram import main
# assign directory
directory = "C:\\Users\\Marcel\\Desktop\\IML\\daps\\daps\\ClipsForCNN\\iphone_livingroom1"
x = 0
# iterate over files in
# that directory
for root, dirs, files in os.walk(directory):
    for filename in files:
        if (not re.match(r'^(f1_|f7|f8|m3|m6|m8)', filename)) and (re.match(r'.*\.wav$', filename)):
            main(root, filename)
            x += 1
            print(filename)
