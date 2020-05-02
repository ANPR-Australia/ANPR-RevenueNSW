import configparser
import os
import yaml
import glob
import alpr
import json
import utils 
import sys
from openalpr import Alpr
import pprint


def run_experiments():
    config = configparser.ConfigParser()
    config.read("config.py")

    prefix = config["DEFAULT"]["prefix"]

    openalpr_runtime = prefix+config["DEFAULT"]["open_alpr_runtime_data"]

    test_data_dir = prefix+config["DEFAULT"]["test_data_dir"]
    labeled_data_dir = prefix+config["DEFAULT"]["labeled_data_dir"]
    config_file_name = config["DEFAULT"]["open_alpr_config_file_name"]
    calibration_files = prefix+config["DEFAULT"]["open_alpr_calibration_dir"]
    results_dir = prefix+config["DEFAULT"]["results"]
    
    dbSchema = config["DB"]["dbSchema"]
    dbFile = config["DB"]["dbFile"]
    dbOld = config["DB"]["dbOld"]

    conn = utils.init_db(dbFile, dbOld, dbSchema)

    label_dict = utils.create_labeled_data_from_images(conn, labeled_data_dir)
    
    cropped_results = test_cropped_number_plates(conn, results_dir, 
            config_file_name, test_data_dir, openalpr_runtime)
    print(cropped_results)
    """

    label_dict = utils.create_labeled_data_from_rnsw_test_data(conn, test_data_dir)
    
    untrained_results = test_untrained_uncalibrated_system(conn, results_dir, 
            config_file_name, test_data_dir, openalpr_runtime)
    evaluationResults = {
                         "matches": 0, 
                         "errors": 0, 
        #                 "dicts": [{}]
                         }

    for camera in untrained_results:
        (matches, errors, eval_dict) = evaluate_results(results_dir, 
                    "untrained_uncalibrated_system", 
                    untrained_results[camera], label_dict)
        evaluationResults["matches"] = evaluationResults["matches"] + matches
        evaluationResults["errors"] = evaluationResults["errors"] + errors 
        #evaluationResults["dicts"] = evaluationResults["dicts"].append(eval_dict)

    pprint.pprint(evaluationResults)

    evaluationResults = {
                         "matches": 0, 
                         "errors": 0, 
        #                 "dicts": [{}]
                         }

    #print("%d percent of number plates detected correctly\n" % matches/len(evaluation_dict))
    calibrated_results = test_untrained_calibrated_system(conn, results_dir, config_file_name, 
            test_data_dir, openalpr_runtime, calibration_files)
    for camera in calibrated_results:
        (matches, errors, eval_dict) = evaluate_results(results_dir, 
                        "untrained_calibrated_system",
                        calibrated_results[camera], label_dict)
    
        evaluationResults["matches"] = evaluationResults["matches"] + matches
        evaluationResults["errors"] = evaluationResults["errors"] + errors 
        #evaluationResults["dicts"] = evaluationResults["dicts"].append(eval_dict)

    pprint.pprint(evaluationResults)
    """
    utils.close_db(conn)


"""
Checks the results against our labeled data. Shows matches and
failures.
"""
def evaluate_results(results_dir, test_name, results_dict, label_dict):
    evaluation_dict = {}
    matches = 0
    errors = 0
    results_file_name = os.path.join(results_dir, "evaluation_results_"+test_name+".json")
    res_out = open(results_file_name, "a+") #open and append to the file


    for f in results_dict:
        r = {}
        file_name = os.path.basename(f)
        expected = label_dict[file_name]
        result = results_dict[file_name]
        r["filename"] = file_name
        r["expected"] = expected
        #compare expected with result
        print("________________________________")
        try:
            r["plate"] = result['results'][0]['plate']
            r["confidence"] = result['results'][0]['confidence']
            if r["plate"]==r["expected"][1]:
                evaluation_dict[file_name] = True
                matches = matches + 1
            else:
                errors = errors + 1
                evaluation_dict[file_name] = False
        except:
            print("no numberplate detected in %s" % file_name)
            errors = errors + 1
            evaluation_dict[file_name] = False
        res_out.write(pprint.pformat(r))
        res_out.write("\n\n")
        print(pprint.pformat(r))

    print("________________________________")
    print("matches: %d" % matches)
    print("errors: %d" % errors)
    print("evaluation_dict: %s" % str(evaluation_dict))
    return (matches, errors, evaluation_dict)



        


"""
Run this test before calliberating the cameras or training
the system. Only run it on half the data available, so it's
consistent with the trained system.

Requires a file called openalpr.conf in the test_data_dir (top level). 
This file should be empty
"""
def test_untrained_uncalibrated_system(conn, results_dir, config_file_name,
        test_data_dir ,openalpr_runtime):
    print("*******test_untrained_uncalibrated_system*****")
    test_name = "test_untrained_uncalibrated_system"
    results = {}
    country_str = "au,auwide"

    loc_dirs = [f.path for f in os.scandir(test_data_dir) if f.is_dir()]
    for loc in loc_dirs:
        camera_dirs = [f.path for f in os.scandir(loc) if f.is_dir()]
        for cam in camera_dirs:
            results[cam] = test_camera(conn, country_str, results_dir, 
                    "test_untrained_calibrated_system", cam, 
                        config_file_name, openalpr_runtime)
        
    return results


"""
Tests all the files in a directory (recursive).
Loads config file from that directory.
If multiple directories for multiple cameras are present, it will
traverse them and aggregate the results. This is how it behaves
for the uncalibrated test.
"""
def test_camera(conn, country_str, results_dir, test_name, test_data_dir, 
        config_file_name, openalpr_runtime, loc=None, 
        cam=None, calibration_files=None): 
    openalpr_conf = os.path.join(test_data_dir, config_file_name)
    results_file_name =  os.path.join(results_dir, test_name+".json")
    if loc:
        openalpr_conf = os.path.join(calibration_files, 
                loc.zfill(8)+"-"+cam+"-prewarp.conf")
        results_file_name =  os.path.join(results_dir, 
                test_name+"_"+loc+"_"+cam+".json")

    print(openalpr_conf)
    results = {}

    alpr = Alpr(country_str, openalpr_conf, openalpr_runtime)
    if not alpr.is_loaded():
        print("Error loading OpenALPR")
        sys.exit(1)

    files = [f for f in glob.glob(test_data_dir + "/*.jpg", recursive=True)]
    results = {}
    for f in files:
        r = alpr.recognize_file(f)
        filename = os.path.basename(f)
        results[filename] = r
        plate = None
        confidence = None
        try:
            plate = r['results'][0]['plate']
            confidence = r['results'][0]['confidence']
        except:
            print("No number plate detected in img %s\n" % filename)

        utils.insert_result(conn, test_name, filename, country_str, config_file_name, plate, confidence, r)

    alpr.unload()

    return results

def test_cropped_number_plates(conn, results_dir, config_file_name, test_data_dir ,openalpr_runtime):
    print("*******test_cropped*****")
    import pdb;pdb.set_trace()
    test_name = "test_cropped_number_plates"
    results = {}
    country_str = "au"

    results[test_data_dir] = test_camera(conn, country_str, results_dir, 
                    "test_cropped", test_data_dir, 
                        config_file_name, openalpr_runtime)
        
    return results

"""
Gets all the directories under our test_data_dir, and assuming each
represents one camera worth of files, it runs the test on those
files using the config file for that particular camera at that location.
data/test/location/camera/<image files>
"""
def test_untrained_calibrated_system(conn, results_dir, config_file_name,test_data_dir, openalpr_runtime, calibration_files):
    print("*****test_untrained_calibrated_system******")
    loc_dirs = [f.path for f in os.scandir(test_data_dir) if f.is_dir()]
    country_str = "au,auwide"
    results = {}
    for loc in loc_dirs:
        camera_dirs = [f.path for f in os.scandir(loc) if f.is_dir()]
        for cam in camera_dirs:
            results[cam] = test_camera(conn, country_str, results_dir, "test_untrained_calibrated_system", cam, 
                        config_file_name, openalpr_runtime, os.path.basename(loc), os.path.basename(cam), calibration_files)
    #print("test_untrained_calib results:")
    #pprint.pprint(results)
    return results


def test_trained_system_no_fonts(test_name, test_data_dir, config_file_name, openalpr_runtime):
    pass


def test_trained_system_with_fonts(test_name, test_data_dir, config_file_name, openalpr_runtime):
    pass


if __name__ == "__main__":
    run_experiments()

