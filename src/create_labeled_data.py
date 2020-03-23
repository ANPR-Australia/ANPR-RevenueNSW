import configparser
import os
import yaml
import glob

config = configparser.ConfigParser()
config.read("config.py")

labeled_data_dir = config["DEFAULT"]["labeled_data_dir"]
print(labeled_data_dir)

labeled_data_output = config["DEFAULT"]["labeled_data_output"]
out = open(labeled_data_output, "a+")

files = [f for f in glob.glob(labeled_data_dir + "/*.yaml", recursive=False)]
print(files)


for f in files:
    with open(f, 'r') as stream:
        try:
            contents = yaml.safe_load(stream)
            out.write("%s,%s,%s\n" % (contents['image_file'], contents['region_code_gt'], contents['plate_number_gt']))
        except yaml.YAMLError as exc:
            print(exc)

out.close()
