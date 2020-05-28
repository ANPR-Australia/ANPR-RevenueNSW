import configparser
import string
import os
import yaml
import glob
import sys
import cv2
import numpy as np
import darknet_detector
import time


def empty_image(image, s=None, error_log=None):
    (H, W) = image.shape[:2]
    if W <= 0 or H <= 0:
        if error_log:
            ps = "{sfile} width={W} height={H}\n"
            formatted_str = ps.format(sfile=s, W=W, H=H)
            error_log.write(formatted_str)
            print(formatted_str)
        return True
    return False


def setup_detector(detector_path, detector_name, darknet_dll=None):
    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([detector_path, detector_name+".names"])
    labels = open(labelsPath).read().strip().split("\n")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([detector_path, detector_name+".weights"])
    configPath = os.path.sep.join([detector_path, detector_name+".cfg"])
    dataPath = os.path.sep.join([detector_path, detector_name+".data"])
    create_data_file(dataPath, labelsPath, len(labels))
    # Load our darknet C++ detector
    if darknet_dll:
        net = darknet_detector.Detector(detector_name, darknet_dll,
                                        configPath, weightsPath,
                                        dataPath, labelsPath)
        return (net, labels)

    # load our YOLO object detector using openCV DNN
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    return (net, labels)


def create_data_file(dataPath, labelsPath, nClasses):
    data_file = open(dataPath, "w+")
    data_file.write("classes = " + str(nClasses))
    data_file.write("names = " + labelsPath)
    data_file.close()


def run_object_detector(name, image, net, labels, min_confidence,
                        threshold, image_name, size=(416, 416)):
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities

    # cv2.imshow("original_img", image)
    # cv2.waitKey(0)
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, size,
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(labels), 3),
                               dtype="uint8")

    # show timing information on YOLO
    print("[INFO] {name} YOLO took {sec} seconds".format(
        name=name, sec=(end - start)))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    centers = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > min_confidence:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                centers.append(box.astype("int"))

    # print(classIDs)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence,
                            threshold)

    cropped_images = []

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            # text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            text = "{}".format(labels[classIDs[i]])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
            vname = image_name + "_" + \
                str(i) + "_" + str(classIDs[i]) + "_" + str(confidences[i])
            if x < 0:
                x = 0
            if y < 0:
                y = 0

            cropped = image[y:y+h, x:x+w]
            cropped_images.append((cropped, vname))

    # show the output image
    return (boxes, confidences, classIDs, cropped_images, centers)


def train_aussie_plates(input_dir, output_dir, detector_path):
    classes = {"nsw": 1,
               "vic": 2,
               "qld": 3,
               "sa":  4,
               "nt":  5,
               "act": 6,
               "wa":  7,
               "tas": 8}
    prepare_yolo_training(input_dir, output_dir, detector_path, classes)


def prepare_yolo_training(input_dir, output_dir, detector_path, classes):
    yaml_files = [f for f in glob.glob(input_dir + "/*.yaml", recursive=False)]
    image_path = os.path.join(output_dir, "img")
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    data_path = os.path.join(output_dir, "data")

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    (yolov3_net, yolov3_labels) = setup_detector(detector_path, "yolov3")
    for y in yaml_files:
        create_yolo_images_and_annotation(y, input_dir, output_dir, classes,
                                          yolov3_net, yolov3_labels, 0.5)
    create_yolo_training_obj_data(output_dir, classes)
    create_yolo_training_obj_names(data_path, classes)
    create_yolo_train_txt(output_dir, data_path)


def create_yolo_images_and_annotation(yaml_file,
                                      input_dir, output_dir,
                                      classes, net, labels,
                                      confidence):
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

        # shutil.copyfile(full_image_path, os.path.join(output_dir,
        #                "img", yaml_obj['image_file']))

        cc = plate_corners.strip().split()
        for i in range(0, len(cc)):
            cc[i] = int(cc[i])

        points = np.array([(cc[0], cc[1]), (cc[2], cc[3]),
                           (cc[4], cc[5]), (cc[6], cc[7])])

        rect = cv2.boundingRect(points)
        (x1, y1, w, h) = rect
        center_of_rect = (x1+w*0.5, y1+h*0.5)

        image = cv2.imread(full_image_path)
        (bxs, confidences, classIDs, candidates, ctrs) = run_object_detector(
                    "yolov3", image, net, labels, confidence,
                    0.25,  yaml_obj['image_file'], (448, 288))

        vehicles = []
        centers = []
        for cl, ve, ct in zip(classIDs, candidates, ctrs):
            # coco classIDs for car, motorbike, truck, boat, bus, train
            if cl in [2, 3, 7, 8, 5, 6]:
                vehicles.append(ve)
                centers.append(ct)
            else:
                print("other classID:")
                print(cl)
        for (ctr, vehicle) in zip(centers, vehicles):
            if empty_image(vehicle[0]):
                continue
            if inside(ctr, center_of_rect):
                new_x = center_of_rect[0] - (ctr[0]-ctr[2]*0.5)
                new_y = center_of_rect[1] - (ctr[1]-ctr[3]*0.5)

                center_of_rect = (new_x, new_y)
                image = vehicle[0]
                break
        # debug
        # img = cv2.imread(full_image_path)
        # color = (255, 0, 0)
        # thickness = 2
        # img = cv2.rectangle(img, (x1, y1), (x1+w, y1+h), color, thickness)

        # cv2.imshow("blah", img)
        # cv2.waitKey(0)
        path = os.path.join(output_dir, "img", yaml_obj['image_file'])
        cv2.imwrite(path, image)
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


def inside(center, point):
    (cx, cy, w, h) = center
    (x, y) = point
    x1 = cx - w*0.5
    y1 = cx - h*0.5
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x >= x1 and y >= y1:
        if x <= cx+w*0.5 and y <= cy+h*0.5:
            return True
    return False


def show_all_yolo_annotation(input_dir, output_dir):
    images = [f for f in glob.glob(input_dir + "/*.jpg", recursive=False)]
    for i in images:
        print(i)
        img = show_yolo_annotation(i)
        path = os.path.join(output_dir, os.path.basename(i))
        print(path)
        cv2.imwrite(path, img)


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
        # cv2.imshow("test", image)
        # cv2.waitKey(0)
        return image


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


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.py")

    labeled_data_dir = config["DEFAULT"]["labeled_data_dir"]
    schema = config["DB"]["dbSchema"]
