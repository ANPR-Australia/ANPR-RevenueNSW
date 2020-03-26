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


    label_dict = create_labeled_data.create_labeled_data(labeled_data_dir, labeled_data_output)

    alpr = Alpr("au", openalpr_conf, openalpr_runtime)
    results = {}
    if not alpr.is_loaded():
        print("Error loading OpenALPR")
        sys.exit(1)


    untrained_results = test_untrained_uncaliberated_system(alpr, test_data_dir)
    (matches, errors, evaluation_dict) = evaluate_results(untrained_results, label_dict)
    #print("%d percent of number plates detected correctly\n" % matches/len(evaluation_dict))






def evaluate_results(results_dict, label_dict):
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
"""
def test_untrained_uncaliberated_system(alpr, test_data_dir):
    files = [f for f in glob.glob(test_data_dir + "/*.jpg", recursive=False)]
    results = {}
    for f in files:
        results[os.path.basename(f)] = alpr.recognize_file(f)
        #print(json.dumps(results, indent=4)) #for debugging only
    alpr.unload()
    return results



        


def test_untrained_caliberated_system(results_dict, label_dict):
    pass


def test_trained_system_no_fonts(results_dict, label_dict):
    pass


def test_trained_system_with_fonts(results_dict, label_dict):
    pass


if __name__ == "__main__":
    run_experiments()

