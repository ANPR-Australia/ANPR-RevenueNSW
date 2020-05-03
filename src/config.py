[DEFAULT]
prefix = /Users/sara/work/
labeled_data_dir =ANPR-RevenueNSW/data/training_rnsw
training_data_dir = ANPR-RevenueNSW/data/training
test_data_dir = ANPR-RevenueNSW/data/training_rnsw

open_alpr_runtime_data = ANPR-RevenueNSW/runtime_data
open_alpr_calibration_dir = ANPR-RevenueNSW/data/calibration
open_alpr_config_file_name = openalpr.conf
results = ANPR-RevenueNSW/data/results

[DB]
dbFile = results.db
dbOld = results_old.db
dbSchema = schema.sql


[PLATE_CROPPER]
input_dir = ANPR-RevenueNSW/data/images
output_dir = ANPR-RevenueNSW/data/cropped_plates
classified_dir = ANPR-RevenueNSW/data/cropped_plates/classified

[YOLO]
input_image_dir = ANPR-RevenueNSW/data/training_rnsw
darknet_model_dir = ANPR-RevenueNSW/data/darknet_data
confidence = 0.5
threshold = 0.3
error_log = ANPR-RevenueNSW/src/yolo_error.log
error_images = ANPR-RevenueNSW/data/yolo_error_images
number_plates = ANPR-RevenueNSW/data/yolo_numberplates
vehicles = ANPR-RevenueNSW/data/yolo_vehicles




