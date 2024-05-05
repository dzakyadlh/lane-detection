import os

image_files = []
os.chdir(os.path.join("data", "valid"))
# x64
# os.chdir(os.path.join("build", "darknet", "x64", "data", "valid"))
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".jpg"):
        image_files.append("data/valid/" + filename)
        # x64
        # image_files.append("build/darknet/x64/data/valid/" + filename)
os.chdir("..")
with open("valid.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("..")