import utils
import configparser


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("config.py")

    prefix = config["DEFAULT"]['prefix']
    input_dir = prefix+config["PLATE_CROPPER"]['input_dir']
    output_dir = prefix+config["PLATE_CROPPER"]['output_dir']
    classified_dir = prefix+config["PLATE_CROPPER"]['classified_dir']

    utils.crop_images(input_dir, output_dir)
    utils.utils.split_into_dirs(classified_dir, input_dir, input_dir)
