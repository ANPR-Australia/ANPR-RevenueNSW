# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import numpy as np
import time
import cv2
import os
import glob
import utils
import darknet_detector


def setup_yolo(conn=None):
    import configparser
    config = configparser.ConfigParser()
    config.read("config.py")

    prefix = config["DEFAULT"]["prefix"]
    image_dir = prefix+config["YOLO"]["input_image_dir"]
    print(image_dir)
    detector_path = prefix+config["YOLO"]["darknet_model_dir"]
    confidence = float(config["YOLO"]["confidence"])
    threshold = float(config["YOLO"]["threshold"])

    error_dir = prefix+config["YOLO"]["error_images"]
    np_dir = prefix+config["YOLO"]["number_plates"]
    vehicle_dir = prefix+config["YOLO"]["vehicles"]
    all_dir = prefix+config["YOLO"]["all_images"]
    darknet_dll = config["YOLO"]["darknet_dll"]
    make_paths([error_dir, np_dir, vehicle_dir, all_dir])

    error_log_file = prefix+config["YOLO"]["error_log"]
    error_log = open(error_log_file, "w+")
    pipeline(image_dir, detector_path, confidence, threshold,
             error_log, error_dir, np_dir, vehicle_dir, all_dir,
             darknet_dll, conn)


def make_paths(path_list):
    for p in path_list:
        if not os.path.exists(p):
            os.makedirs(p)


def pipeline(image_dir, detector_path, confidence, threshold,
             error_log, error_dir, np_dir, vehicle_dir, all_dir,
             darknet_dll, conn=None):
    (vd_net, vd_labels) = setup_detector(detector_path, "vehicle-detection")
    (lpd_net, lpd_labels) = setup_detector(
        detector_path, "lp-detection-layout-classification")
    (lpr_net, lpr_labels) = setup_detector(detector_path, "lp-recognition")
    (yolov3_net, yolov3_labels) = setup_detector(detector_path, "yolov3")
    darknet_lpd, _ = setup_detector(detector_path,
                                    "lp-detection-layout-classification",
                                    darknet_dll)
    darknet_lpr, _ = setup_detector(detector_path,
                                    "lp-recognition",
                                    darknet_dll)

    images = [f for f in glob.glob(image_dir + "/*.jpg")]
    images.sort()
    for img in images:
        image_fname = os.path.basename(img)
        image_name = os.path.splitext(image_fname)[0]
        print(img)
        # load our input image and grab its spatial dimensions
        image = cv2.imread(img)
        if empty_image(image, img, error_log):
            utils.insert_result(conn, "yolo", image_fname,
                                "au", "empty_image", None, -1, "")
            continue
        (boxes, confidences, classIDs, vehicles) = run_object_detector(
            "vd", image, vd_net, vd_labels, confidence, threshold, image_name,
            (448, 288))
        if len(classIDs) == 0:
            utils.insert_result(conn, "yolo", image_fname, "au",
                                "no_vehicle_detected_vehicle_net", None, -1,
                                "")
            # try it with yolov3
            (boxes, confidences, classIDs, candidates) = run_object_detector(
                "yolov3", image, yolov3_net, yolov3_labels, confidence,
                threshold, image_name, (448, 288))
            vehicles = []
            for cl, ve in zip(classIDs, candidates):
                # coco classIDs for car, motorbike, truck, boat, bus, train
                if cl in [2, 3, 7, 8, 5, 6]:
                    vehicles.append(ve)
                else:
                    print("other classID:")
                    print(cl)
            if len(classIDs) == 0:
                utils.insert_result(
                    conn, "yolo", image_fname, "au",
                    "no_vehicle_detected_yolov3_net", None, -1, "")
                cv2.imwrite(os.path.join(error_dir, image_fname), image)

        for (vehicle, v_name) in vehicles:
            if empty_image(vehicle,
                           "vehicle_of_" + str(len(vehicles)) + "_" +
                           v_name, error_log):
                # here we could continue the pipeline with image, assuming
                # the reason it can't find the vehicle is
                # because we're zoomed in too much.. worth testing this idea.
                utils.insert_result(conn, "yolo", os.path.basename(
                    img), "au", "malformed_vehicle_detected", None,
                    len(vehicles), "")
                # (boxes, confidences, classIDs, lps) =
                # run_object_detector(image, lpd_net, lpd_labels, confidence,
                # 0.1, v_name)
                cv2.imwrite(os.path.join(error_dir, image_fname), image)
                vehicle = image  # EXPERIMENTAL
            cv2.imwrite(os.path.join(vehicle_dir, image_fname), vehicle)
            (boxes, confidences, classIDs, lps) = run_object_detector(
                "lpd", vehicle, lpd_net, lpd_labels, confidence, 0.1, v_name)
            # cv2.imshow("vehicle", vehicle)
            # cv2.waitKey(0)

            if len(classIDs) == 0:
                utils.insert_result(conn, "yolo", image_fname,
                                    "au", "no_lp_detected", None, -1, "")
                cv2.imwrite(os.path.join(error_dir, image_fname), vehicle)
                (boxes, confidences, classIDs, lps) = \
                    darknet_lpd.run_object_detector(
                                os.path.join(vehicle_dir, image_fname),
                                thresh=0.1, obj_name=v_name)
                if len(classIDs) == 0:
                    utils.insert_result(conn, "yolo", image_fname,
                                        "au", "no_lp_detected_darknet",
                                        None, -1, "")

            for (lp, lp_name) in lps:
                if empty_image(lp, "plate_"+lp_name, error_log):
                    utils.insert_result(
                        conn, "yolo", image_fname, "au",
                        "malformed_lp_detected", None, len(lps), "")
                    cv2.imwrite(os.path.join(error_dir, image_fname), vehicle)
                    lp = vehicle  # EXPERIMENTAL
                (boxes, confidences, classIDs, plate_contents) = \
                    run_object_detector("lpr", lp, lpr_net, lpr_labels,
                                        confidence, 0.5, lp_name, (352, 128))
                if len(classIDs) == 0:
                    utils.insert_result(
                        conn, "yolo", image_fname, "au",
                        "no_characters_recognised", None, -1, "")
                    cv2.imwrite(os.path.join(error_dir, image_fname), lp)

                    resized = cv2.resize(image, (352, 128))
                    np_path = os.path.join(np_dir, image_fname)
                    cv2.imwrite(np_path, resized)
                    (boxes, confidences, classIDs, lps) = \
                        darknet_lpr.run_object_detector(
                                    np_path,
                                    thresh=0.5, obj_name=lp_name)
                if len(classIDs) == 0:
                    utils.insert_result(
                        conn, "yolo", image_fname, "au",
                        "no_characters_recognised_darknet", None, -1, "")
                else:
                    # sort the characters based on x value, then
                    # join them all up into a numberplate
                    number_plate = "".join(
                        [lpr_labels[bc[1]] for bc
                            in sorted(zip(boxes, classIDs),
                                      key=get_x)])
                    print(number_plate)
                    if conn:
                        utils.insert_result(
                            conn, "yolo", image_fname, "au",
                            "number_plate_recognised", number_plate, -1, "")
                cv2.imwrite(os.path.join(np_dir, image_fname), lp)
        cv2.imwrite(os.path.join(all_dir, image_fname), image)


def empty_image(image, s, error_log):
    (H, W) = image.shape[:2]
    if W <= 0 or H <= 0:
        ps = "{sfile} width={W} height={H}\n"
        formatted_str = ps.format(sfile=s, W=W, H=H)
        error_log.write(formatted_str)
        print(formatted_str)
        return True
    return False


def get_x(box):
    return box[0]


def create_data_file(dataPath, labelsPath, nClasses):
    data_file = open(dataPath, "w+")
    data_file.write("classes = " + str(nClasses))
    data_file.write("names = " + labelsPath)
    data_file.close()


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
        net = darknet_detector.Detector(darknet_dll,
                                        configPath, weightsPath,
                                        dataPath, labelsPath)
        return (net, labels)

    # load our YOLO object detector using openCV DNN
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    return (net, labels)


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
            cropped = image[y:y+h, x:x+w]
            cropped_images.append((cropped, vname))

    # show the output image
    return (boxes, confidences, classIDs, cropped_images)


if __name__ == "__main__":
    setup_yolo()
