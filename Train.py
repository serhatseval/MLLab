# import required module
import os
import re
from Plot_Spectrograms import main

# assign directory
directory = "AudioFiles"
x = 0
# iterate over files in
# that directory

for root, dirs, files in os.walk(directory):
    for filename in files:
        if (re.match(r'^(f1_|f7|f8|m3|m6|m8)', filename)) and (re.match(r'.*\.wav$', filename)) and not filename.startswith((".", "_")):
            main(root, filename, 1)
            x += 1
            print(filename)