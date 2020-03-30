import configparser
import os
import yaml
import glob
import alpr
import json
import create_labeled_data
import sys
from openalpr import Alpr


def run_experiments():
    config = configparser.ConfigParser()
    config.read("config.py")

    openalpr_conf = config["DEFAULT"]["open_alpr_config"]
    openalpr_runtime = config["DEFAULT"]["open_alpr_runtime_data"]

    test_data_dir = config["DEFAULT"]["test_data_dir"]
    labeled_data_dir = config["DEFAULT"]["labeled_data_dir"]
    labeled_data_output = config["DEFAULT"]["labeled_data_output"]
    config_file_name = config["DEFAULT"]["open_alpr_config_file_name"]


    label_dict = create_labeled_data.create_labeled_data(labeled_data_dir, labeled_data_output)

    untrained_results = test_untrained_uncaliberated_system(config_file_name, test_data_dir)
    (matches, errors, evaluation_dict) = evaluate_results(untrained_results, label_dict)
    #print("%d percent of number plates detected correctly\n" % matches/len(evaluation_dict))

    caliberated_results = test_untrained_caliberated_system(config_file_name, test_data_dir)
    for camera in caliberated_results:
        evaluate_results(camera, caliberated_results[camera], label_dict)
    
"""
Checks the results against our labeled data. Shows matches and
failures.

Asghar: can you please write this to a file too?
"""
def evaluate_results(test_name, results_dict, label_dict):
    evaluation_dict = {}
    matches = 0
    errors = 0

    for f in results_dict:
        file_name = os.path.basename(f)
        expected = label_dict[file_name]
        result = results_dict[file_name]
        #compare expected with result
        print("--------")
        try:
            plate = result['results'][0]['plate']
            confidence = result['results'][0]['confidence']
            print(plate)
            print(confidence)
            print(expected)
            if plate==expected[1]:
                evaluation_dict[file_name] = True
                matches = matches + 1
            else:
                errors = errors + 1
                evaluation_dict[file_name] = False
        except:
            print("no numberplate detected in %s" % file_name)
            errors = errors + 1
            evaluation_dict[file_name] = False

    print("___________________")
    print(matches)
    print(errors)
    print(evaluation_dict)
    return (matches, errors, evaluation_dict)



        


"""
Run this test before calliberating the cameras or training
the system. Only run it on half the data available, so it's
consistent with the trained system.

Requires a file called openalpr.conf in the test_data_dir (top level). 
This file should be empty
"""
def test_untrained_uncaliberated_system(test_data_dir, config_file_name, openalpr_runtime):
    test_name = "test_untrained_uncaliberated_system"
    test_camera(test_name, test_data_dir, config_file_name, openalpr_runtime)


"""
Tests all the files in a directory (recursive).
Loads config file from that directory.
If multiple directories for multiple cameras are present, it will
traverse them and aggregate the results. This is how it behaves
for the uncaliberated test.
"""
def test_camera(test_name, test_data_dir, config_file_name, openalpr_runtime): 
    openalpr_conf = os.path.join(test_data_dir, config_file_name)
    alpr = Alpr("au", openalpr_conf, openalpr_runtime)
    results = {}
    results_file_name =  os.path.join(test_data_dir, results_file_name)
    res_out = open(results_file_name, "w") #open and truncate the file
    res_out.write("[")

    if not alpr.is_loaded():
        print("Error loading OpenALPR")
        sys.exit(1)

    files = [f for f in glob.glob(test_data_dir + "/*.jpg", recursive=True)]
    results = {}
    for f in files:
        results[os.path.basename(f)] = alpr.recognize_file(f)
        res_out.write(json.dumps(results, indent=4)) #for debugging only
        res_out.write(",\n")
    alpr.unload()
    res_out.write("]")
    return results

"""
Gets all the directories under our test_data_dir, and assuming each
represents one camera worth of files, it runs the test on those
files using the config file in the dir.
"""
def test_untrained_caliberated_system(test_name, test_data_dir, config_file_name):
    camera_dirs = [f.path for f in os.scandir(test_data_dir) if f.is_dir()]
    results = {}
    for cam in camera_dirs:
        results[cam] = test_camera(test_name, test_data_dir, config_file_name, openalpr_runtime)

    return results


def test_trained_system_no_fonts(results_dict, label_dict):
    pass


def test_trained_system_with_fonts(results_dict, label_dict):
    pass


if __name__ == "__main__":
    run_experiments()

