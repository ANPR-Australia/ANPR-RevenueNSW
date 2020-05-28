# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import cv2
import os
import glob
import utils
import pytesseract
import yolo_utils as yutils
from pytesseract import Output


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
    # recognise_plates(image_dir, detector_path, confidence, threshold,
    pipeline(image_dir, detector_path, confidence, threshold,
             error_log, error_dir, np_dir, vehicle_dir, all_dir,
             darknet_dll, conn)


def make_paths(path_list):
    for p in path_list:
        if not os.path.exists(p):
            os.makedirs(p)


def recognise_tesserect(image_path, min_conf):
    # load the input image, convert it from BGR to RGB channel ordering,
    # and use Tesseract to localize each area of text in the input image
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pytesseract.image_to_string(rgb, output_type=Output.DICT)

    # loop over each of the individual text localizations
    for i in range(0, len(results["text"])):
        # extract the bounding box coordinates of the text region from
        # the current result
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]

        # extract the OCR text itself along with the confidence of the
        # text localization
        text = results["text"][i]
        conf = int(results["conf"][i])

        # filter out weak confidence text localizations
        if conf > min_conf:
            # display the confidence and text to our terminal
            print("Confidence: {}".format(conf))
            print("Text: {}".format(text))
            print("")

            # strip out non-ASCII text so we can draw the text on the image
            # using OpenCV, then draw a bounding box around the text along
            # with the text itself
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)

    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)


def recognise_plates(image_dir, detector_path, confidence, threshold,
                     error_log, error_dir, np_dir, vehicle_dir, all_dir,
                     darknet_dll, conn=None):
    (lpr_net, lpr_labels) = yutils.setup_detector(detector_path,
                                                  "lp-recognition")
    darknet_lpr, _ = yutils.setup_detector(detector_path,
                                           "lp-recognition",
                                           darknet_dll)

    images = [f for f in glob.glob(image_dir + "/*.jpg")]
    images.sort()
    n_images = len(images)
    i = 0
    for img in images:
        i = i+1
        print("Processing img ("+str(i)+"/"+str(n_images)+")")
        image_fname = os.path.basename(img)
        lp_name = os.path.splitext(image_fname)[0]
        print(img)
        # load our input image and grab its spatial dimensions
        lp = cv2.imread(img)
        if empty_image(lp, img, error_log):
            utils.insert_result(conn, "yolo", image_fname,
                                "au", "empty_image", None, -1, "")
            continue
        resized = cv2.resize(lp, (352, 128))
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
            np_path = os.path.join(np_dir, image_fname)
            cv2.imwrite(np_path, resized)
            (boxes, confidences, classIDs, plate_contents) = \
                yutils.run_object_detector("lpr", lp, lpr_net, lpr_labels,
                                           confidence, 0.5,
                                           lp_name, (352, 128))
        if len(classIDs) == 0:
            utils.insert_result(
                conn, "yolo", image_fname, "au",
                "no_characters_recognised_dnn", None, -1, "")
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


def pipeline(image_dir, detector_path, confidence, threshold,
             error_log, error_dir, np_dir, vehicle_dir, all_dir,
             darknet_dll, conn=None):
    (vd_net, vd_labels) = yutils.setup_detector(detector_path,
                                                "vehicle-detection")
    (lpd_net, lpd_labels) = yutils.setup_detector(
        detector_path, "lp-detection-layout-classification")
    (lpr_net, lpr_labels) = yutils.setup_detector(detector_path,
                                                  "lp-recognition")
    (yolov3_net, yolov3_labels) = yutils.setup_detector(detector_path,
                                                        "yolov3")
    darknet_lpd, _ =\
        yutils.setup_detector(detector_path,
                              "lp-detection-layout-classification",
                              darknet_dll)
    darknet_lpr, _ =\
        yutils.setup_detector(detector_path, "lp-recognition",
                              darknet_dll)

    images = [f for f in glob.glob(image_dir + "/*.jpg")]
    images.sort()
    n_images = len(images)
    i = 0
    for img in images:
        i = i+1
        print("Processing img ("+str(i)+"/"+str(n_images)+")")
        image_fname = os.path.basename(img)
        image_name = os.path.splitext(image_fname)[0]
        print(img)
        # load our input image and grab its spatial dimensions
        image = cv2.imread(img)
        if empty_image(image, img, error_log):
            utils.insert_result(conn, "yolo", image_fname,
                                "au", "empty_image", None, -1, "")
            continue
        (boxes, confidences, classIDs, vehicles) = yutils.run_object_detector(
            "vd", image, vd_net, vd_labels, confidence, 0.25, image_name,
            (448, 288))
        if len(classIDs) == 0:
            utils.insert_result(conn, "yolo", image_fname, "au",
                                "no_vehicle_detected_vehicle_net", None, -1,
                                "")
            # try it with yolov3
            (boxes, confidences, classIDs, candidates) =\
                yutils.run_object_detector(
                "yolov3", image, yolov3_net, yolov3_labels, confidence,
                0.25, image_name, (448, 288))
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
                # Assume we're cropped in too much to detect a vehicle and
                # try finding the number plate in the whole image.
                vehicles = (image, "whole_image")

        for (vehicle, v_name) in vehicles:
            if empty_image(vehicle,
                           "vehicle_of_" + str(len(vehicles)) + "_" +
                           v_name, error_log):
                utils.insert_result(conn, "yolo", os.path.basename(
                    img), "au", "malformed_vehicle_detected", None,
                    len(vehicles), "")
                # cv2.imwrite(os.path.join(error_dir, image_fname), image)
                # here we could continue the pipeline with image, assuming
                # the reason it can't find the vehicle is
                # because we're zoomed in too much.. worth testing this idea.
                vehicle = image  # EXPERIMENTAL
            cv2.imwrite(os.path.join(vehicle_dir, image_fname), vehicle)
            (boxes, confidences, classIDs, lps) = \
                darknet_lpd.yutils.run_object_detector(
                                os.path.join(vehicle_dir, image_fname),
                                thresh=0.1, obj_name=v_name)

            if len(classIDs) == 0:
                utils.insert_result(conn, "yolo", image_fname,
                                    "au", "no_lp_detected_darknet",
                                    None, -1, "")
                (boxes, confidences, classIDs, lps) =\
                    yutils.run_object_detector(
                    "lpd", vehicle, lpd_net, lpd_labels,
                    confidence, 0.1, v_name)

                if len(classIDs) == 0:
                    utils.insert_result(conn, "yolo", image_fname,
                                        "au", "no_lp_detected_dnn",
                                        None, -1, "")

            for (lp, lp_name) in lps:
                if empty_image(lp, "plate_"+lp_name, error_log):
                    utils.insert_result(
                        conn, "yolo", image_fname, "au",
                        "malformed_lp_detected", None, len(lps), "")
                    cv2.imwrite(os.path.join(error_dir, image_fname), vehicle)
                    lp = vehicle  # EXPERIMENTAL

                resized = cv2.resize(lp, (352, 128))
                np_path = os.path.join(np_dir, image_fname)
                cv2.imwrite(np_path, resized)
                (boxes, confidences, classIDs, lps) = \
                    darknet_lpr.yutils.run_object_detector(
                                    np_path,
                                    thresh=0.5, obj_name=lp_name)
                if len(classIDs) == 0:
                    utils.insert_result(
                        conn, "yolo", image_fname, "au",
                        "no_characters_recognised_darknet", None, -1, "")
                    np_path = os.path.join(np_dir, image_fname)
                    cv2.imwrite(np_path, resized)
                    (boxes, confidences, classIDs, plate_contents) = \
                        yutils.run_object_detector("lpr", lp,
                                                   lpr_net, lpr_labels,
                                                   confidence, 0.5,
                                                   lp_name, (352, 128))
                if len(classIDs) == 0:
                    utils.insert_result(
                        conn, "yolo", image_fname, "au",
                        "no_characters_recognised_dnn", None, -1, "")
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
    return yutils.empty_image(image, s, error_log)


def get_x(box):
    return box[0]


if __name__ == "__main__":
    setup_yolo()
