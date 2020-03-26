# ANPR-RevenueNSW
Exploring ANPR solutions for number plates in Australia

You need to compile openalpr from source, and know the location of the source files for the pip install below.

```
python3 -m venv alpr
source alpr/bin/activate

python -m pip install --editable <$location_of_openalpr_source_code/src/bindings/python>
python -m pip install --editable .   

```
