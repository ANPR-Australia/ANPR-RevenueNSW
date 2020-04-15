CREATE TABLE IF NOT EXISTS "labels"
(
	image_file_name TEXT PRIMARY KEY NOT NULL,
	region_code TEXT NOT NULL,
	plate_number TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS "results"
(
	image_file_name TEXT NOT NULL,
	test_name TEXT NOT NULL,
	country_str TEXT NOT NULL,
	openalpr_conf_file TEXT NOT NULL,
	first_plate TEXT,
	confidence REAL,
	json_str TEXT
);


CREATE TABLE IF NOT EXISTS "file_metadata"
(
	image_file_name PRIMARY_KEY_NOT_NULL,
	capture_date DATE NOT NULL,
	cameraType TEXT NOT NULL,
	location_id TEXT NOT NULL,
	incident_id TEXT NOT NULL,
	camera_id TEXT NOT NULL
);

