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

