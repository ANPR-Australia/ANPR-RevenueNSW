import configparser
import os
import yaml
import glob
import sys
import shutil
import sqlite3


"""
Filename is in the format of:
date     cameraType locationID IncidentID.CameraID.jpg
YYYYMMDD XX         DDDDDDDD  ZZZZZZZZ    CCC     .jpg
for example:
20200205060002856000000001.001.jpg

date: 2020/02/05
cameraType: 06
locationID = 00028560
incidentID = 00000001
cameraID = 001

returns a dictionary with incident metadata in it.
"""
def parse_filename(filename):
    r = {}
    r['date'] = filename[0:8]
    r['cameraType'] = filename[8:10]
    r['locationID'] = filename[10:18]
    r['incidentID'] = filename[18:26]
    r['cameraID'] = filename[27:30]
    return r


def put_in_directories(pooled_data_dir, destination_dir, file_type):
    all_files = [f for f in glob.glob(pooled_data_dir + "/*."+file_type, recursive=False)]
    for f in all_files:
        bits = parse_filename(os.path.basename(f))
        new_location = os.path.join(destination_dir, bits["locationID"], bits["cameraID"])
        print(new_location)
        print(f)
        if not os.path.exists(new_location):
            os.makedirs(new_location)
        shutil.copyfile(f, os.path.join(new_location, os.path.basename(f)))



def create_labeled_data(conn, labeled_data_dir, labeled_data_output):
    label_dict = {}
    files = []

    loc_dirs = [f.path for f in os.scandir(labeled_data_dir) if f.is_dir()]
    for loc in loc_dirs:
        camera_dirs = [f.path for f in os.scandir(loc) if f.is_dir()]
        for cam in camera_dirs:
            new_files = [f for f in glob.glob(cam + "/*.yaml", recursive=False)]
            files = files + new_files

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
                    sys.exit(1) ;
                insert_label(conn, img_file, region_code, plate_no)
                insert_metadata(conn, img_file)
                label_dict[os.path.basename(img_file)] = (region_code, plate_no)
            except yaml.YAMLError as exc:
                print(exc)
                system.exit(1)

    return label_dict



def crop_images(input_dir, out_dir):
    """
    Yaml files are generated using the plate_tagger utility in openALPR.
    This function reads the yaml file, and crops the plate out of the image.
    """

    if not os.path.isdir(input_dir):
        print("input_dir (%s) doesn't exist" % input_dir)
        sys.exit(1)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    yaml_files = []

    yaml_files = [f for f in glob.glob(input_dir + "/*.yaml", recursive=False)]
    yaml_files.sort()

    count = 1
    for yaml_file in yaml_files:
        print("Processing: " + yaml_file + " (" + str(count) + "/" + str(len(yaml_files)) + ")")
        yaml_path = os.path.join(input_dir, yaml_file)
        yaml_without_ext = os.path.splitext(yaml_path)[0]
        
        yaml_obj = yaml.safe_load(stream)
        
        # Skip missing images
        full_image_path = os.path.join(input_dir, yaml_obj['image_file'])
        if not os.path.isfile(full_image_path):
            print("Could not find image file %s, skipping" % (full_image_path))
            continue


        plate_corners = yaml_obj['plate_corners_gt']
        cc = plate_corners.strip().split()
        for i in range(0, len(cc)):
            cc[i] = int(cc[i])

        img = cv2.imread(full_image_path)
        mask = np.zeros(img.shape[0:2], dtype=np.uint8)
        points = np.array([[[cc[0],cc[1]],[cc[2], cc[3]], [cc[4],cc[5]], [cc[6],cc[7]]]])

        cv2.drawContours(mask, [points], -1, (255,255,255), -1, cv2.LINE_AA)

        res = cv2.bitwise_and(img,img,mask = mask)
        rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
        cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
        out_crop_path = os.path.join(out_dir, os.path.basename(yaml_without_ext) + ".jpg")
        cv2.imwrite(out_crop_path, cropped )
        count += 1

    print("%d Cropped images are located in %s" % (count-1, out_dir))


def init_db(dbFile, dbOld, dbSchema):
    """
    Truncates old database files, and loads the schema into
    a fresh database. Returns db connector.
    """
    #if there's already a file there, cp it before
    #truncating it to store new values
    if os.path.exists(dbFile):
        print("Moving and truncating old database file")
        shutil.copy(dbOld, dbFile)
        f = open(dbFile, "a")
        f.truncate(0)
        f.close()
    
    conn = sqlite3.connect(dbFile)
    with open(dbSchema) as fp:
        conn.executescript(fp.read())
    print("Loaded schema")

    return conn

def close_db(conn):
    conn.close()

def insert_label(conn, img_file, region_code, plate_no):
    c = conn.cursor()
    values = (img_file, region_code, plate_no)
    c.execute('''INSERT INTO labels (image_file_name, region_code, plate_number) VALUES (?, ?, ?)''', values)
    conn.commit()


def insert_metadata(conn, img_file):
    c = conn.cursor()
    r = parse_filename(img_file)
    values = (img_file, r['date'], r['cameraType'], r['locationID'], r['incidentID'], r['cameraID'])
    c.execute('''INSERT INTO file_metadata (image_file_name, capture_date, cameraType, location_id, incident_id, camera_id) values (?,?,?,?,?,?)''', values)
    conn.commit()



def insert_result(conn, test_name, img_file_name,
        country_str, openalpr_conf_file, first_plate, confidence,
        json_str):
    c = conn.cursor()
    if first_plate:
        values = (img_file_name, test_name, country_str, openalpr_conf_file,
                first_plate, confidence, str(json_str))
        print(values)
        c.execute('''INSERT INTO results (image_file_name, test_name, country_str, openalpr_conf_file, first_plate, confidence, json_str) VALUES (?, ?,?,?,?,?,?)''', values)
    else:
        values =  (img_file_name, test_name, country_str, openalpr_conf_file)
        c.execute('''INSERT INTO results (image_file_name, test_name, country_str, openalpr_conf_file) VALUES (?,?,?,?)''', values)

    conn.commit()
        


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.py")

    labeled_data_dir = config["DEFAULT"]["labeled_data_dir"]
    labeled_data_output = config["DEFAULT"]["labeled_data_output"]
    schema = config["DB"]["dbSchema"]

    conn = init_db("util_test.db", "util_test_old.db", schema)

    print(create_labeled_data(conn, labeled_data_dir, labeled_data_output))
