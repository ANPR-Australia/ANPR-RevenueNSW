import configparser
import os
import yaml
import glob
import alpr
import json
import create_labeled_data


def run_experiments():
    config = configparser.ConfigParser()
    config.read("config.py")

    openalpr_conf = config["DEFAULT"]["open_alpr_config"]
    openalpr_runtime = config["DEFAULT"]["open_alpr_runtime_data"]

    test_data_dir = config["DEFAULT"]["test_data_dir"]
    label_dict = create_labeled_data()

    alpr = ("au", openalpr_conf, openalpr_runtime)
    results = {}
    if not alpr.is_loaded()
        error("Error loading OpenALPR")


    untrained_results = test_untrained_caliberated_system(alpr, test_data_dir)
    evaluation_dict = evaluate_results(untrained_results, label_dict)






def evaluate_results(results_dict, label_dict):
    evaluation_dict = {}
    for file_name in results_dict:
        expected = label_dict[file_name]
        result = results_dict[file_name]
        #compare expected with result
        if result.something == expected: #you've got to run it and see what the fieldnames in the json object are
            evaluation_dict[file_name] = True
        else
            evaluation_dict[file_name] = False #you can directly assign the result of the comparison, but I left it like this so you can see what the result is more easily while coding, will fix it later.

    return evaluation_dict



        


"""
Run this test before calliberating the cameras or training
the system. Only run it on half the data available, so it's
consistent with the trained system.
"""
def test_untrained_uncaliberated_system(alpr, test_data_dir):
    files = [f for f in glob.glob(test_data_dir + "/*.jpg", recursive=False)]
    for f in files:
        results[f] = alpr.recognize_file(f)
        print(json.dumps(results, indent=4)) #for debugging only
    alpr.unload()



        


def test_untrained_caliberated_system():
    pass


def test_trained_system_no_fonts():
    pass


def test_trained_system_with_fonts():
    pass


if __name__ == "__main__:
    run_experiments()

