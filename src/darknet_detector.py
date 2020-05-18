#!python3
"""
Python 3 wrapper for identifying objects in images

Requires DLL compilation

Both the GPU and no-GPU version should be compiled; the no-GPU version should
be renamed "yolo_cpp_dll_nogpu.dll".

On a GPU system, you can force CPU evaluation by any of:

- Set global variable DARKNET_FORCE_CPU to True
- Set environment variable CUDA_VISIBLE_DEVICES to -1
- Set environment variable "FORCE_CPU" to "true"


To use, either run performDetect() after import, or modify the end of this
file.

See the docstring of performDetect() for parameters.

Directly viewing or returning bounding-boxed images requires scikit-image to be
installed (`pip install scikit-image`)

@author: Philip Kahn
@date: 20180503
"""
from ctypes import (
        CDLL, Structure, c_float, c_int, POINTER, c_char_p, RTLD_GLOBAL,
        c_void_p, pointer)
import os
import cv2
import numpy as np

"""
def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

"""


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int)]


class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int),
                ("dets", POINTER(DETECTION))]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


class Detector():

    def network_width(self, net):
        return self.lib.network_width(net)

    def network_height(self, net):
        return self.lib.network_height(net)

    def __init__(self, dll_path, configPath, weightPath, dataPath, labelsPath):
        """
            dll_path: str
                should be where libdarknet.so lives.

            configPath: str
                Path to the configuration file. Raises ValueError if not found

            weightPath: str
                Path to the weights file. Raises ValueError if not found

            dataPath: str
                Path to the data file. Raises ValueError if not found

            labelsPath: str
                Path to labels file

        """
        self.hasGPU = True
        self.lib = CDLL(dll_path, RTLD_GLOBAL)

        self.lib.network_width.argtypes = [c_void_p]
        self.lib.network_width.restype = c_int
        self.lib.network_height.argtypes = [c_void_p]
        self.lib.network_height.restype = c_int

        self.copy_image_from_bytes = self.lib.copy_image_from_bytes
        self.copy_image_from_bytes.argtypes = [IMAGE, c_char_p]

        self.predict = self.lib.network_predict_ptr
        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        if self.hasGPU:
            self.set_gpu = self.lib.cuda_set_device
            self.set_gpu.argtypes = [c_int]

        # init_cpu = self.lib.init_cpu

        self.make_image = self.lib.make_image
        self.make_image.argtypes = [c_int, c_int, c_int]
        self.make_image.restype = IMAGE

        self.get_network_boxes = self.lib.get_network_boxes
        self.get_network_boxes.argtypes = [
            c_void_p,
            c_int,
            c_int,
            c_float,
            c_float,
            POINTER(c_int),
            c_int,
            POINTER(c_int),
            c_int]
        self.get_network_boxes.restype = POINTER(DETECTION)

        self.make_network_boxes = self.lib.make_network_boxes
        self.make_network_boxes.argtypes = [c_void_p]
        self.make_network_boxes.restype = POINTER(DETECTION)

        self.free_detections = self.lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.free_batch_detections = self.lib.free_batch_detections
        self.free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

        self.free_ptrs = self.lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.network_predict = self.lib.network_predict_ptr
        self.network_predict.argtypes = [c_void_p, POINTER(c_float)]

        self.reset_rnn = self.lib.reset_rnn
        self.reset_rnn.argtypes = [c_void_p]

        self.load_net = self.lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.load_net_custom = self.lib.load_network_custom
        self.load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
        self.load_net_custom.restype = c_void_p

        self.do_nms_obj = self.lib.do_nms_obj
        self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.do_nms_sort = self.lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.free_image = self.lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.letterbox_image = self.lib.letterbox_image
        self.letterbox_image.argtypes = [IMAGE, c_int, c_int]
        self.letterbox_image.restype = IMAGE

        self.load_meta = self.lib.get_metadata
        self.lib.get_metadata.argtypes = [c_char_p]
        self.lib.get_metadata.restype = METADATA

        self.load_image = self.lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.rgbgr_image = self.lib.rgbgr_image
        self.rgbgr_image.argtypes = [IMAGE]

        self.predict_image = self.lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)

        self.predict_image_letterbox = self.lib.network_predict_image_letterbox
        self.predict_image_letterbox.argtypes = [c_void_p, IMAGE]
        self.predict_image_letterbox.restype = POINTER(c_float)

        self.network_predict_batch = self.lib.network_predict_batch
        self.network_predict_batch.argtypes = [
            c_void_p,
            IMAGE,
            c_int,
            c_int,
            c_int,
            c_float,
            c_float,
            POINTER(c_int),
            c_int,
            c_int]
        self.network_predict_batch.restype = POINTER(DETNUMPAIR)

        if not os.path.exists(configPath):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(configPath) + "`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(weightPath) + "`")
        if not os.path.exists(dataPath):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(dataPath) + "`")

        self.net = self.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
        self.meta = self.load_meta(dataPath.encode("ascii"))

        # In Python 3, the metafile default access craps out on Windows (but
        # not Linux) Read the names file and create a list to feed to detect

        if os.path.exists(labelsPath):
            with open(labelsPath) as namesFH:
                namesList = namesFH.read().strip().split("\n")
                self.altNames = [x.strip() for x in namesList]

        print("Initialized detector")

    def array_to_image(self, arr):
        # need to return old values to avoid python freeing memory
        arr = arr.transpose(2, 0, 1)
        c = arr.shape[0]
        h = arr.shape[1]
        w = arr.shape[2]
        arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
        data = arr.ctypes.data_as(POINTER(c_float))
        im = IMAGE(w, h, c, data)
        return im, arr

    def classify(self, im):
        out = self.predict_image(self.net, im)
        res = []
        for i in range(self.meta.classes):
            if self.altNames is None:
                nameTag = self.meta.names[i]
            else:
                nameTag = self.altNames[i]
            res.append((nameTag, out[i]))
        res = sorted(res, key=lambda x: -x[1])
        return res

    def detect(self, image, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
        """
        Performs the meat of the detection
        """
        im = self.load_image(image, 0, 0)
        if debug:
            print("Loaded image")
        ret = self.detect_image(im, thresh, hier_thresh, nms, debug)
        self.free_image(im)
        if debug:
            print("freed image")
        return ret

    def run_object_detector(
            self,
            image,
            obj_name,
            thresh=.5,
            hier_thresh=.5,
            nms=.45,
            debug=False):
        print("***** using C++ detector *****")
        # custom_image_bgr = cv2.imread(image) # use: detect(,,imagePath,)
        # custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
        # custom_image = cv2.resize(custom_image,(lib.network_width(net),
        #     lib.network_height(net)), interpolation = cv2.INTER_LINEAR)
        # custom_image = scipy.misc.imread(image)
        # im, arr = array_to_image(custom_image)          # you should comment
        # line below: self.free_image(im)
        im = self.load_image(image.encode("ascii"), 0, 0)
        num = c_int(0)
        if debug:
            print("Assigned num")
        pnum = pointer(num)
        if debug:
            print("Assigned pnum")
        self.predict_image(self.net, im)
        letter_box = 0
        # self.predict_image_letterbox(net, im)
        # letter_box = 1
        if debug:
            print("did prediction")
        # dets = get_network_boxes(net, custom_image_bgr.shape[1],
        #    custom_image_bgr.shape[0],
        #    thresh, hier_thresh, None, 0, pnum, letter_box) # OpenCV
        # self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float,
        #    c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
        dets = self.get_network_boxes(
            self.net,
            im.w,
            im.h,
            thresh,
            hier_thresh,
            None,
            0,
            pnum,
            letter_box)
        if debug:
            print("Got dets")
        num = pnum[0]
        if debug:
            print("got zeroth index of pnum")
        if nms:
            self.do_nms_sort(dets, num, self.meta.classes, nms)
        if debug:
            print("did sort")
        res = []
        if debug:
            print("about to range")
        boxes = []
        confidences = []
        classIDs = []
        cropped_images = []
        for j in range(num):
            if debug:
                print("Ranging on " + str(j) + " of " + str(num))
            if debug:
                print("Classes: " + str(self.meta),
                      self.meta.classes, self.meta.names)
            for i in range(self.meta.classes):
                if debug:
                    print("Class-ranging on " + str(i) + " of " +
                          str(self.meta.classes) + "= " + str(dets[j].prob[i]))
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    if self.altNames is None:
                        nameTag = self.meta.names[i]
                    else:
                        nameTag = self.altNames[i]
                    if debug:
                        print("Got bbox", b)
                        print(nameTag)
                        print(dets[j].prob[i])
                        print((b.x, b.y, b.w, b.h))
                    res.append(
                        (nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
                    boxes.append([b.x, b.y, b.w, b.h])
                    confidences.append(dets[j].prob[i])
                    classIDs.append(i)
                    cv_img = cv2.imread(image)
                    (x, y, w, h) = self.bounding_box((b.x, b.y, b.w, b.h))
                    cropped = cv_img[y:y+h, x:x+w]
                    vname = obj_name + "_" + \
                        str(i) + "_" + nameTag + "_" + str(dets[j].prob[i])

                    cropped_images.append((cropped, vname))
        if debug:
            print("did range")
        res = sorted(res, key=lambda x: -x[1])
        if debug:
            print("did sort")
        self.free_detections(dets, num)
        if debug:
            print("freed detections")
        # return res
        return (boxes, confidences, classIDs, cropped_images)

    def bounding_box(self, bounds):
        yExtent = int(bounds[3])
        xExtent = int(bounds[2])
        # Coordinates are around the center
        xCoord = int(bounds[0] - bounds[2] / 2)
        yCoord = int(bounds[1] - bounds[3] / 2)
        """
        boundingBox = [
            [xCoord, yCoord],
            [xCoord, yCoord + yExtent],
            [xCoord + xEntent, yCoord + yExtent],
            [xCoord + xEntent, yCoord]
        ]
        """
        return (xCoord, yCoord, xExtent, yExtent)
        # return boundingBox

    def detect_image(
            self,
            im,
            thresh=.5,
            hier_thresh=.5,
            nms=.45,
            debug=False):
        # custom_image_bgr = cv2.imread(image) # use: detect(,,imagePath,)
        # custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
        # custom_image = cv2.resize(custom_image,(lib.network_width(net),
        #     lib.network_height(net)), interpolation = cv2.INTER_LINEAR)
        # custom_image = scipy.misc.imread(image)
        # im, arr = array_to_image(custom_image)          # you should comment
        # line below: self.free_image(im)
        num = c_int(0)
        if debug:
            print("Assigned num")
        pnum = pointer(num)
        if debug:
            print("Assigned pnum")
        self.predict_image(self.net, im)
        letter_box = 0
        # self.predict_image_letterbox(net, im)
        # letter_box = 1
        if debug:
            print("did prediction")
        # dets = get_network_boxes(net, custom_image_bgr.shape[1],
        #    custom_image_bgr.shape[0],
        #    thresh, hier_thresh, None, 0, pnum, letter_box) # OpenCV
        # self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float,
        #    c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
        dets = self.get_network_boxes(
            self.net,
            im.w,
            im.h,
            thresh,
            hier_thresh,
            None,
            0,
            pnum,
            letter_box)
        if debug:
            print("Got dets")
        num = pnum[0]
        if debug:
            print("got zeroth index of pnum")
        if nms:
            self.do_nms_sort(dets, num, self.meta.classes, nms)
        if debug:
            print("did sort")
        res = []
        if debug:
            print("about to range")
        for j in range(num):
            if debug:
                print("Ranging on " + str(j) + " of " + str(num))
            if debug:
                print("Classes: " + str(self.meta),
                      self.meta.classes, self.meta.names)
            for i in range(self.meta.classes):
                if debug:
                    print("Class-ranging on " + str(i) + " of " +
                          str(self.meta.classes) + "= " + str(dets[j].prob[i]))
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    if self.altNames is None:
                        nameTag = self.meta.names[i]
                    else:
                        nameTag = self.altNames[i]
                    if debug:
                        print("Got bbox", b)
                        print(nameTag)
                        print(dets[j].prob[i])
                        print((b.x, b.y, b.w, b.h))
                    res.append(
                        (nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        if debug:
            print("did range")
        res = sorted(res, key=lambda x: -x[1])
        if debug:
            print("did sort")
        self.free_detections(dets, num)
        if debug:
            print("freed detections")
        return res

    def performDetect(
            self,
            imagePath,
            thresh=0.25,
            showImage=True,
            makeImageOnly=False):
        """
        Convenience function to handle the detection and returns of objects.

        Displaying bounding boxes requires libraries scikit-image and numpy

        Parameters
        ----------------
        imagePath: str
            Path to the image to evaluate. Raises ValueError if not found

        thresh: float (default= 0.25)
            The detection threshold

        showImage: bool (default= True)
            Compute (and show) bounding boxes. Changes return.

        makeImageOnly: bool (default= False)
            If showImage is True, this won't actually *show* the image,
            but will create the array and return it.


        Returns
        ----------------------


        When showImage is False, list of tuples like
            ('obj_label', confidence, (bounding_box_x_px, bounding_box_y_px,
            bounding_box_width_px, bounding_box_height_px))
            The X and Y coordinates are from the center of the bounding box.
            Subtract half the width or height to get the lower corner.

        Otherwise, a dict with
            {
                "detections": as above
                "image": a numpy array representing an image,
                    compatible with scikit-image
                "caption": an image caption
            }
        """
        print("**** Using C++ Darknet Detector *****")
        if not os.path.exists(imagePath):
            raise ValueError("Invalid image path `" +
                             os.path.abspath(imagePath) + "`")
        # Do the detection
        # detections = detect(netMain, meta, imagePath, thresh)   # if is used
        # cv2.imread(image)
        detections = self.detect(imagePath.encode("ascii"), thresh)
        if showImage:
            try:
                from skimage import io, draw
                image = io.imread(imagePath)
                print("*** " + str(len(detections)) +
                      " Results, color coded by confidence ***")
                imcaption = []
                for detection in detections:
                    label = detection[0]
                    confidence = detection[1]
                    pstring = label + ": " + \
                        str(np.rint(100 * confidence)) + "%"
                    imcaption.append(pstring)
                    print(pstring)
                    bounds = detection[2]
                    shape = image.shape
                    # x = shape[1]
                    # xExtent = int(x * bounds[2] / 100)
                    # y = shape[0]
                    # yExtent = int(y * bounds[3] / 100)
                    yExtent = int(bounds[3])
                    xEntent = int(bounds[2])
                    # Coordinates are around the center
                    xCoord = int(bounds[0] - bounds[2] / 2)
                    yCoord = int(bounds[1] - bounds[3] / 2)
                    boundingBox = [
                        [xCoord, yCoord],
                        [xCoord, yCoord + yExtent],
                        [xCoord + xEntent, yCoord + yExtent],
                        [xCoord + xEntent, yCoord]
                    ]
                    # Wiggle it around to make a 3px border
                    rr, cc = draw.polygon_perimeter(
                            [x[1] for x in boundingBox],
                            [x[0] for x in boundingBox],
                            shape=shape)
                    rr2, cc2 = draw.polygon_perimeter(
                        [x[1] + 1 for x in boundingBox],
                        [x[0] for x in boundingBox], shape=shape)
                    rr3, cc3 = draw.polygon_perimeter(
                        [x[1] - 1 for x in boundingBox],
                        [x[0] for x in boundingBox], shape=shape)
                    rr4, cc4 = draw.polygon_perimeter(
                            [x[1] for x in boundingBox],
                            [x[0] + 1 for x in boundingBox], shape=shape)
                    rr5, cc5 = draw.polygon_perimeter(
                            [x[1] for x in boundingBox],
                            [x[0] - 1 for x in boundingBox], shape=shape)
                    boxColor = (int(255 * (1 - (confidence ** 2))),
                                int(255 * (confidence ** 2)), 0)
                    draw.set_color(image, (rr, cc), boxColor, alpha=0.8)
                    draw.set_color(image, (rr2, cc2), boxColor, alpha=0.8)
                    draw.set_color(image, (rr3, cc3), boxColor, alpha=0.8)
                    draw.set_color(image, (rr4, cc4), boxColor, alpha=0.8)
                    draw.set_color(image, (rr5, cc5), boxColor, alpha=0.8)
                if not makeImageOnly:
                    io.imshow(image)
                    io.show()
                detections = {
                    "detections": detections,
                    "image": image,
                    "caption": "\n<br/>".join(imcaption)
                }
            except Exception as e:
                print("Unable to show image: " + str(e))
        return detections

    def performBatchDetect(
            self,
            thresh=0.25,
            hier_thresh=.5,
            nms=.45,
            batch_size=3):
        # NB! Image sizes should be the same
        # You can change the images, yet, be sure that they have the same width
        # and height
        img_samples = ['data/person.jpg', 'data/person.jpg', 'data/person.jpg']
        image_list = [cv2.imread(k) for k in img_samples]

        pred_height, pred_width, c = image_list[0].shape
        net_width, net_height = (
            self.network_width(self.net), self.network_height(self.net))
        img_list = []
        for custom_image_bgr in image_list:
            custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
            custom_image = cv2.resize(
                custom_image, (net_width, net_height),
                interpolation=cv2.INTER_NEAREST)
            custom_image = custom_image.transpose(2, 0, 1)
            img_list.append(custom_image)

        arr = np.concatenate(img_list, axis=0)
        arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
        data = arr.ctypes.data_as(POINTER(c_float))
        im = IMAGE(net_width, net_height, c, data)

        batch_dets = self.network_predict_batch(
            self.net,
            im,
            batch_size,
            pred_width,
            pred_height,
            thresh,
            hier_thresh,
            None,
            0,
            0)
        batch_boxes = []
        batch_scores = []
        batch_classes = []
        for b in range(batch_size):
            num = batch_dets[b].num
            dets = batch_dets[b].dets
            if nms:
                self.do_nms_obj(dets, num, self.meta.classes, nms)
            boxes = []
            scores = []
            classes = []
            for i in range(num):
                det = dets[i]
                score = -1
                label = None
                for c in range(det.classes):
                    p = det.prob[c]
                    if p > score:
                        score = p
                        label = c
                if score > thresh:
                    box = det.bbox
                    left, top, right, bottom = map(
                        int, (box.x - box.w / 2, box.y - box.h / 2,
                              box.x + box.w / 2, box.y + box.h / 2))
                    boxes.append((top, left, bottom, right))
                    scores.append(score)
                    classes.append(label)
                    boxColor = (int(255 * (1 - (score ** 2))),
                                int(255 * (score ** 2)), 0)
                    cv2.rectangle(image_list[b], (left, top),
                                  (right, bottom), boxColor, 2)
            cv2.imwrite(os.path.basename(img_samples[b]), image_list[b])

            batch_boxes.append(boxes)
            batch_scores.append(scores)
            batch_classes.append(classes)
        self.free_batch_detections(batch_dets, batch_size)
        return batch_boxes, batch_scores, batch_classes


if __name__ == "__main__":
    detector = Detector("./libdarknet.so", "./cfg/yolov4.cfg",
                        "yolov4.weights", "./cfg/coco.data")
    detector.performDetect("./data/car.jpg", 0.25)
    detector = Detector("./libdarknet.so", "./cfg/yolov4.cfg",
                        "yolov4.weights", "./cfg/coco.data",
                        "./cfg/coco.names")
    detector.performDetect("./data/car.jpg", 0.25)
