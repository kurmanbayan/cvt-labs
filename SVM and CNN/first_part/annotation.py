import os

annotation_file = open("annotation.txt", "w")

for root, dirs, files in os.walk("pos"):
    for file in files:
        annotation_file.write(file + ' pos\n')

for root, dirs, files in os.walk("neg"):
    for file in files:
        annotation_file.write(file + ' neg\n')

annotation_file.close()
