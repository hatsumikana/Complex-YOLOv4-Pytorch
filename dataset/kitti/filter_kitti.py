import argparse
import pathlib
import os
import logging
import shutil


# Parse Arguments for Dynamic File Reads
parser = argparse.ArgumentParser()

parser.add_argument("-o", "--output", help = "Output Location for Copied Dataset", required=True)
parser.add_argument("-s", "--src_folder", help = "Dataset Location", required = True)
parser.add_argument("-l", "--log", help = "Set Logging level (None/DEBUG)")

args = parser.parse_args()

if args.log == 'DEBUG':
    logging.basicConfig(level = logging.DEBUG)
else :
    logging.basicConfig(level = logging.INFO)

# Verify File Location
out_folder_path = pathlib.Path(args.output)
src_folder_path = pathlib.Path(args.src_folder)
label_folder = src_folder_path.joinpath("label_2")

if not src_folder_path.is_dir():
    raise FileNotFoundError("Not a valid source folder")
elif not label_folder.is_dir():
    raise FileNotFoundError("Not a valid source folder")

global_count = {"Car": 0, "Pedestrian": 0, "Cyclist": 0}
valid_idx = []
subfolders = ["label_2","calib","image_2","velodyne"]

# Read File and extract labels
logging.info("Retrieving Dataset Labels")
label_files = [file for file in label_folder.glob("**"+os.sep+"*") if file.is_file()]

for lfile in label_files:
    local_count = {"Car": 0, "Pedestrian": 0, "Cyclist": 0}
    logging.debug("File Opening: "+ lfile.name)
    with open(lfile, mode='r') as fis:
        for line in fis:
            class_label = line.split(" ")[0]
            if class_label in local_count.keys():
                local_count[class_label] = local_count[class_label]+1
                #global_count[class_label] = global_count[class_label]+1
    logging.debug("File Count:")
    logging.debug(local_count)

    if local_count["Pedestrian"] == 0 and local_count["Cyclist"] == 0:
        logging.debug("File Rejected")
    else: 
        for key in global_count.keys():
            global_count[key] += local_count[key]
        valid_idx.append(lfile.stem)
logging.info("Global Class Label Count:")
logging.info(global_count)
logging.info("Data Retrieval Complete. Copying Files...")
logging.debug(valid_idx)

# Copy Files to output
for sub in subfolders:
    suboutput = out_folder_path.joinpath(sub)
    subsource = src_folder_path.joinpath(sub)
    if subsource.is_dir:
        match sub:
            case "label_2":
                suffix = ".txt"
            case "calib":
                suffix = ".txt"
            case "image_2":
                suffix = ".png"
            case "velodyne":
                suffix = ".bin"
        logging.info("Copying Folder "+sub)
    else:
        logging.info("Folder not found: "+sub+". Continuing...")
        continue
    if not suboutput.exists():
        os.makedirs(suboutput)
    for idx in valid_idx:
        shutil.copy(subsource.joinpath(idx+suffix), suboutput)
logging.info("Copy Files Completed. Process End.")





