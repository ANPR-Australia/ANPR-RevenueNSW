import configparser
import os
import yaml
import glob
import sys

"""
Filename is in the format of:
date     cameraType locationID IncidentID.CameraID.jpg
YYYYMMDD XX         DDDDDDDD  ZZZZZZZZ    CCC     .jpg
for example:
20200205060002856000000001.001.jpg

date: 2020/02/05
cameraType: 06
locationID = 60002856
incidentID = 000000001
cameraID = 001

returns a dictionary with incident metadata in it.
"""
def parse_filename(filename):
    r = {}
    r['date'] = filename[0:8]
    r['cameraType'] = filename[8:10]
    r['locationID'] = filename[10:17]
    r['incidentID'] = filename[17:26]
    r['cameraID'] = filename[27:30]
    return r

def create_labeled_data(labeled_data_dir, labeled_data_output):
    out = open(labeled_data_output, "w") #open and truncate the file
    label_dict = {}
    files = [f for f in glob.glob(labeled_data_dir + "/*.yaml", recursive=False)]


    for f in files:
        with open(f, 'r') as stream:
            try:
                try:
                    contents = yaml.safe_load(stream)
                    img_file = contents['image_file']
                    region_code = contents['region_code_gt']
                    plate_no = contents['plate_number_gt']
                except KeyError as e:
                    print("Missing key in file: " + f + "\n" + str(e))
                    sys.exit(1)

                out.write("%s,%s,%s\n" % (img_file, region_code, plate_no))
                label_dict[os.path.basename(img_file)] = (region_code, plate_no)
            except yaml.YAMLError as exc:
                print(exc)
                system.exit(1)

    out.close()
    return label_dict

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.py")

    labeled_data_dir = config["DEFAULT"]["labeled_data_dir"]
    labeled_data_output = config["DEFAULT"]["labeled_data_output"]


    print(create_labeled_data(labeled_data_dir, labeled_data_output))
