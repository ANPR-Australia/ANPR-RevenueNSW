import configparser
import string
import os
import yaml
import glob
import sys
import shutil
import sqlite3
import cv2
import prespective
import numpy as np


def parse_filename(filename):
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
    r = {}
    r['date'] = filename[0:8]
    r['cameraType'] = filename[8:10]
    r['locationID'] = filename[10:18]
    r['incidentID'] = filename[18:26]
    r['cameraID'] = filename[27:30]
    return r


def put_in_directories(pooled_data_dir, destination_dir, file_type):
    all_files = [f for f in glob.glob(
        pooled_data_dir + "/*."+file_type, recursive=False)]
    for f in all_files:
        bits = parse_filename(os.path.basename(f))
        new_location = os.path.join(
            destination_dir, bits["locationID"], bits["cameraID"])
        print(new_location)
        print(f)
        if not os.path.exists(new_location):
            os.makedirs(new_location)
        shutil.copyfile(f, os.path.join(new_location, os.path.basename(f)))


def create_labeled_data_from_rnsw_test_data(conn, labeled_data_dir):
    """
    Read the labels from the yaml files into the DB.
    """
    label_dict = {}
    files = []
    """
    loc_dirs = [f.path for f in os.scandir(labeled_data_dir) if f.is_dir()]
    for loc in loc_dirs:
        camera_dirs = [f.path for f in os.scandir(loc) if f.is_dir()]
        for cam in camera_dirs:
            new_files = [f for f in glob.glob(cam + "/*.yaml",
                recursive=False)]
            files = files + new_files
    """
    files = [f for f in glob.glob(
        labeled_data_dir + "/*.yaml", recursive=False)]

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
                insert_label(conn, img_file, region_code, plate_no)
                insert_metadata(conn, img_file)
                label_dict[os.path.basename(img_file)] = (
                    region_code, plate_no)
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(1)

    return label_dict


def create_labeled_data_from_images(conn, labeled_data_dir):
    """
    Read the labels from the images dir, and write to the db
    """
    label_dict = {}
    files = [f for f in glob.glob(
        labeled_data_dir + "/*.yaml", recursive=False)]
    for f in files:
        with open(f, 'r') as stream:
            try:
                try:
                    contents = yaml.safe_load(stream)
                    img_file = contents['image_file']
                    region_code = "NSW"
                    plate_no = contents['plate_number_gt']
                except KeyError as e:
                    print("Missing key in file: " + f + "\n" + str(e))
                    sys.exit(1)
                split_file = os.path.splitext(os.path.basename(f))
                ext = os.path.splitext(img_file)[1]
                image_file_name = split_file[0]+ext

                insert_label(conn, image_file_name, region_code, plate_no)
                # insert_metadata(conn, img_file)
                label_dict[os.path.basename(img_file)] = (
                    region_code, plate_no)
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(1)

    return label_dict


def rename_files(input_dir, out_dir):
    """
    This renames all the files and associated yaml files into
    more readable filenames, it also generates a mapping between old file
    names and new file names so moved files can be renamed too.
    """
    yaml_files = [f for f in glob.glob(input_dir + "/*.yaml", recursive=False)]
    yaml_files.sort()

    log_path = os.path.join(out_dir, "log.txt")
    logfile = open(log_path, "w")
    log = {}

    count = 1
    for yaml_file in yaml_files:
        with open(yaml_file, 'r') as stream:
            try:
                print("Processing: " + yaml_file +
                      " (" + str(count) + "/" + str(len(yaml_files)) + ")")
                yaml_path = os.path.join(input_dir, yaml_file)

                yaml_obj = yaml.safe_load(stream)
                original_filename = yaml_obj['image_file']
                plate_no = yaml_obj['plate_number_gt']

                new_jpg_filename = plate_no+"_"+str(count)+".jpg"
                yaml_obj['image_file'] = new_jpg_filename

                new_yaml_filename = plate_no+"_"+str(count)+".yaml"

                # now write the new files:
                jpg_path = os.path.join(out_dir, new_jpg_filename)
                yaml_path = os.path.join(out_dir, new_yaml_filename)
                original_path = os.path.join(input_dir, original_filename)

                log[original_filename] = new_jpg_filename

                shutil.copyfile(original_path, jpg_path)
                yf = open(yaml_path, "w")
                yf.write(yaml.dump(yaml_obj))

                count = count+1

            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(1)

    logfile.write(yaml.dump(log))


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
        print("Processing: " + yaml_file +
              " (" + str(count) + "/" + str(len(yaml_files)) + ")")
        crop_image(input_dir, out_dir, yaml_file)
        count = count+1

    print("%d Cropped images are located in %s" % (count-1, out_dir))


def crop_image(input_dir, out_dir, yaml_file):
    with open(yaml_file, 'r') as stream:
        try:
            yaml_path = os.path.join(input_dir, yaml_file)
            yaml_without_ext = os.path.splitext(yaml_path)[0]

            yaml_obj = yaml.safe_load(stream)

            try:
                # Skip missing images
                plate_corners = yaml_obj['plate_corners_gt']
                full_image_path = os.path.join(
                    input_dir, yaml_obj['image_file'])
            except KeyError as e:
                print("Missing key in file: " + yaml_file + "\n" + str(e))
                sys.exit(1)

            if not os.path.isfile(full_image_path):
                print("Could not find image file %s, skipping" %
                      (full_image_path))
                return

            cc = plate_corners.strip().split()
            for i in range(0, len(cc)):
                cc[i] = int(cc[i])

            img = cv2.imread(full_image_path)
            points = np.array([(cc[0], cc[1]), (cc[2], cc[3]),
                               (cc[4], cc[5]), (cc[6], cc[7])])
            cropped = prespective.four_point_transform(img, points)
            out_crop_path = os.path.join(
                out_dir, os.path.basename(yaml_without_ext) + ".jpg")
            print(out_crop_path)
            cv2.imwrite(out_crop_path, cropped)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)


def train_aussie_plates(input_dir, output_dir):
    classes = {"nsw": 1,
               "vic": 2,
               "qld": 3,
               "sa":  4,
               "nt":  5,
               "act": 6,
               "wa":  7,
               "tas": 8}
    prepare_yolo_training(input_dir, output_dir, classes)


def prepare_yolo_training(input_dir, output_dir, classes):
    yaml_files = [f for f in glob.glob(input_dir + "/*.yaml", recursive=False)]
    image_path = os.path.join(output_dir, "img")
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    data_path = os.path.join(output_dir, "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    for y in yaml_files:
        create_yolo_images_and_annotation(y, input_dir, output_dir, classes)
    create_yolo_training_obj_data(output_dir, classes)
    create_yolo_training_obj_names(data_path, classes)
    create_yolo_train_txt(output_dir, data_path)


def create_yolo_images_and_annotation(yaml_file,
                                      input_dir, output_dir,
                                      classes):
    """
    <object-class> <x_center> <y_center> <width> <height>
    """
    yaml_path = os.path.join(input_dir, yaml_file)
    yaml_path = yaml_file
    with open(yaml_path, 'r') as stream:
        yaml_obj = yaml.safe_load(stream)

        try:
            # Skip missing images
            plate_corners = yaml_obj['plate_corners_gt']
            full_image_path = os.path.join(
                input_dir, yaml_obj['image_file'])
            (fn, ext) = os.path.splitext(os.path.basename(full_image_path))
            object_class = yaml_obj['region_code_gt']

        except KeyError as e:
            print("Missing key in file: " + yaml_file + "\n" + str(e))
            sys.exit(1)

        if not os.path.isfile(full_image_path):
            print("Could not find image file %s, skipping" %
                  (full_image_path))
            return

        shutil.copyfile(full_image_path, os.path.join(output_dir,
                        "img", yaml_obj['image_file']))

        cc = plate_corners.strip().split()
        for i in range(0, len(cc)):
            cc[i] = int(cc[i])

        points = np.array([(cc[0], cc[1]), (cc[2], cc[3]),
                           (cc[4], cc[5]), (cc[6], cc[7])])

        rect = cv2.boundingRect(points)
        (x1, y1, w, h) = rect
        # debug
        # img = cv2.imread(full_image_path)
        # color = (255, 0, 0)
        # thickness = 2
        # img = cv2.rectangle(img, (x1, y1), (x1+w, y1+h), color, thickness)

        # cv2.imshow("blah", img)
        # cv2.waitKey(0)
        center_of_rect = (x1+w*0.5, y1+h*0.5)
        s = string.Template("$object_class $x_center $y_center $width $height")
        line = s.substitute(object_class=classes[object_class],
                            x_center=center_of_rect[0],
                            y_center=center_of_rect[1],
                            width=w,
                            height=h)
        output_text_path = os.path.join(output_dir, "img", fn+".txt")
        f = open(output_text_path, "w+")
        f.write(line)
        f.close()


def show_yolo_annotation(image_path):
    image = cv2.imread(image_path)
    txt_path = os.path.splitext(image_path)[0]+".txt"
    with open(txt_path, "r") as f:
        content = f.readlines()
        values = content[0].strip().split()
        for i in range(0, len(values)):
            values[i] = float(values[i])

        # cls = values[0]
        x_centre = values[1]
        y_centre = values[2]
        w = values[3]
        h = values[4]

        x1 = int(x_centre - 0.5*w)
        x2 = int(x_centre + 0.5*w)

        y1 = int(y_centre - 0.5*h)
        y2 = int(y_centre + 0.5*h)

        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        # Displaying the image
        cv2.imshow("test", image)
        cv2.waitKey(0)


def create_yolo_training_obj_data(output_dir, classes):
    s = string.Template("classes=$n_classes\ntrain=data/train.txt" +
                        "\nvalid=data/train.txt\n" +
                        "names=data/obj.names\nbackup = backup/")
    path = os.path.join(output_dir, "obj.data")
    f = open(path, "w+")
    f.write(s.substitute(n_classes=len(classes)))
    f.close()


def create_yolo_training_obj_names(output_dir, classes):
    path = os.path.join(output_dir, "obj.names")
    f = open(path, "w+")
    for c in classes:
        f.write(c+"\n")
    f.close()


def create_yolo_train_txt(output_dir, input_dir):
    path = os.path.join(output_dir, "train.txt")
    f = open(path, "w+")
    images = [f for f in glob.glob(input_dir + "/*.jpg", recursive=False)]
    for i in images:
        f.write(os.path.join(output_dir, "img", i)+"\n")
    f.close()


def capture_visual_classification(classified_dir, rename_log):
    """
    If the cropped images were visually classified by a human, but their
    filenames are all messy, we can capture the classifications and link
    them back to their original filename.

    We should be able to get rid of the yucky rename stuff as soon as
    we've got consistant directories in git. It's a very ugly hack, but it
    worked!
    """
    with open(rename_log, 'r') as stream:
        try:
            renames = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)

    conn = init_db("classification.db", "classification_old.db",
                   "classification.sql")
    classes = [f.path for f in os.scandir(classified_dir) if f.is_dir()]
    for class_name in classes:
        images = [f for f in glob.glob(class_name + "/*.jpg", recursive=False)]
        for image in images:
            print(image)
            split = os.path.basename(image).split("-")
            old_name = "-".join(split[:-1])
            print(old_name)
            new_name = renames[old_name]+".jpg"

            insert_classification(conn, new_name, os.path.basename(class_name))

    close_db(conn)


def split_into_dirs(classified_dir, original_dir, yaml_dir=None):
    """
    Reads the sql database, and splits the files into directories
    based on their classification. This can be used for training
    the OCR system, or for evaluating OCR on cropped images. The filenames
    being consistent means we can compare the metadata and see if our
    OCR was successful.

    Specify a dir with yaml files in it if you want the numberplates
    cropped as well.
    """
    conn = init_db("classification.db", None, "classification.sql")
    rows = get_classifications(conn)
    for (img, d) in rows:
        path = os.path.join(classified_dir, d)
        new_file = os.path.join(path, img)
        old_file = os.path.join(original_dir, img)
        if not os.path.exists(path):
            os.makedirs(path)
        if yaml_dir:
            (fn, ext) = os.path.splitext(img)
            yaml_file = os.path.join(yaml_dir, fn+".yaml")

            crop_image(original_dir, path, yaml_file)
        else:
            shutil.copyfile(old_file, new_file)


def init_db(dbFile, dbOld, dbSchema):
    """
    Truncates old database files, and loads the schema into
    a fresh database. Returns db connector.
    """
    # if there's already a file there, cp it before
    # truncating it to store new values
    if os.path.exists(dbFile):
        if dbOld:
            # don't do it if there's no backup file
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


def insert_classification(conn, name, class_name):
    c = conn.cursor()
    values = (name, class_name)
    c.execute(
        '''
        INSERT INTO classifications (image_file_name, classification)
        VALUES (?,?)
        ''', values)
    conn.commit()


def get_classifications(conn):
    c = conn.cursor()
    c.execute("select * from classifications")
    rows = c.fetchall()
    return rows


def insert_label(conn, img_file, region_code, plate_no):
    c = conn.cursor()
    values = (img_file, region_code, plate_no)
    c.execute(
        '''
        INSERT INTO labels (image_file_name, region_code, plate_number)
        VALUES (?, ?, ?)
        ''', values)
    conn.commit()


def insert_metadata(conn, img_file):
    c = conn.cursor()
    r = parse_filename(img_file)
    values = (img_file, r['date'], r['cameraType'],
              r['locationID'], r['incidentID'], r['cameraID'])
    c.execute('''
            INSERT INTO file_metadata (image_file_name, capture_date,
            cameraType, location_id, incident_id, camera_id)
            VALUES (?,?,?,?,?,?)
            ''', values)
    conn.commit()


def insert_result(conn, test_name, img_file_name,
                  country_str, openalpr_conf_file, first_plate, confidence,
                  json_str):
    c = conn.cursor()
    if first_plate:
        values = (img_file_name, test_name, country_str, openalpr_conf_file,
                  first_plate, confidence, str(json_str))
        print(values)
        c.execute('''
                INSERT INTO results (image_file_name, test_name, country_str,
                openalpr_conf_file, first_plate, confidence, json_str)
                VALUES (?, ?,?,?,?,?,?)''', values)
    else:
        values = (img_file_name, test_name, country_str, openalpr_conf_file)
        c.execute(
            '''
            INSERT INTO results
            (image_file_name, test_name, country_str, openalpr_conf_file)
            VALUES (?,?,?,?)
            ''', values)

    conn.commit()


def replace_all(replacements, string):
    if not string:
        return ""
    for a, b in replacements:
        string = string.replace(a, b)
    return string


def results_by_incident(conn):
    c = conn.cursor()
    c.execute(
        """
         SELECT file_metadata.image_file_name, file_metadata.location_id,
         file_metadata.incident_id, openalpr_conf_file, results.first_plate,
         labels.plate_number
         FROM labels
         INNER JOIN results on results.image_file_name = labels.image_file_name
         INNER JOIN file_metadata
         ON file_metadata.image_file_name = results.image_file_name
         WHERE openalpr_conf_file = "number_plate_recognised"
         GROUP BY location_id, incident_id
         ORDER BY file_metadata.location_id;
         """
    )
    rows = c.fetchall()
    same_count = 0
    replace_same_count = 0
    for res in rows:
        print(res)
        numberplate = res[5]
        result = res[4]
        replacements = [('0', 'O'), ('1', 'I'), ('B', '8'), ('G', '6')]
        if result == numberplate:
            same_count = same_count+1

        if replace_all(replacements, result) == replace_all(replacements,
                                                            numberplate):
            replace_same_count = replace_same_count+1

    result = "same: {same_count}/{total} numberplates the same\n"
    print(result.format(same_count=same_count, total=len(rows)))

    result = "subs: {replace_same_count}/{total} numberplates the same\n"
    print(result.format(replace_same_count=replace_same_count,
          total=len(rows)))


def results_overall(conn):
    correct_dict = {}
    c = conn.cursor()
    c.execute(
        """
         SELECT labels.image_file_name, openalpr_conf_file,
         results.first_plate, labels.plate_number
         FROM labels
         INNER JOIN results ON
         results.image_file_name = labels.image_file_name;
        """
    )
    rows = c.fetchall()
    same_count = 0
    replace_same_count = 0
    for res in rows:
        print(res)
        filename = res[0]
        numberplate = res[3].upper()
        result = res[2]
        if not result:
            continue
        result = result.upper()
        replacements = [('0', 'O'), ('1', 'I'), ('B', '8'), ('G', '6')]
        if result.upper() == numberplate.upper():
            same_count = same_count+1
            correct_dict[filename] = result

        if replace_all(replacements, result) == replace_all(replacements,
                                                            numberplate):
            replace_same_count = replace_same_count+1
            correct_dict[filename] = result

    result = "same: {same_count}/{total} numberplates the same\n"
    print(result.format(same_count=same_count, total=len(rows)))

    result = "subs: {replace_same_count}/{total} numberplates the same\n"
    print(result.format(replace_same_count=replace_same_count,
          total=len(rows)))
    print("number correct overall")
    print(len(correct_dict))


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.py")

    labeled_data_dir = config["DEFAULT"]["labeled_data_dir"]
    schema = config["DB"]["dbSchema"]

    conn = init_db("util_test.db", "util_test_old.db", schema)
