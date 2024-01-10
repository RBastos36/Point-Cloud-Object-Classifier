import glob


# Get filenames of all images (including sub-folders)
dataset_filenames = glob.glob('data/objects_pcd/**/*.pcd', recursive=True)

# Check if dataset data exists
if len(dataset_filenames) < 1:
    raise FileNotFoundError('Dataset files not found')


for file_name in dataset_filenames:

    inputFile = open(file_name, "r")            # Input-file
    data = inputFile.read().splitlines(True)

    new_file_name = file_name.replace(".pcd", ".pts")
    new_file_name = new_file_name.split("/")[-1]
    new_file_name = "/home/rantonio/Desktop/savi_t2/data/objects_pts/" + new_file_name

    print(new_file_name)

    outputFile = open(new_file_name, "w") # Output-file

    data = data[10:]

    for line in data:
        line_data = line.split(" ")
        outputFile.write(str(line_data[0]) + " " + str(line_data[1]) + " " + str(line_data[2]) + "\n")

    inputFile.close()
    outputFile.close()

print("All done")
