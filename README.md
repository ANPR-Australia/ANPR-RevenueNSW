# ANPR-RevenueNSW
Exploring ANPR solutions for number plates in Australia

You need to compile openalpr from source, and know the location of the source files for the pip install below.

```
python3 -m venv alpr
source alpr/bin/activate

python -m pip install --editable <$location_of_openalpr_source_code/src/bindings/python>
python -m pip install --editable .   

```

For the purposes of pattern matching, the alpha-numeric formats and the
corresponding alpr patterns are mentioned below for standard plates. These do
not include any custom plates.

New South Wales (Taken from: https://www.myplates.com.au/)

1. AA-99-AA nsw @@##@@
2. AAA-99A nsw @@@###@
3. AAA-999 nsw @@@###
4. AA-999 nsw @@###
5. AA-9999 nsw @@####
6. N-AA99A nsw N@@##@
7. N-AA999 nsw N@@###
10. 999-AAA nsw ###@@@
11. 99-AAAA nsw ##@@@@
12. 99-AAA nsw ##@@@

Tasmania (Taken from https://tasplates.com)

  Cars:
  1. AAA-999 tas @@@###
  2. AAA-99 tas @@@##
  3. AA-999 tas @@###
  4. T-AA999 tas [T]@@###
  5. Open content: 1-6 characters with numbers and letters
  6. Numeral only: 1-6 numbers

  Motorbikes:
  1. AA-999 @@###
  2. AAA-99 @@@##
  3. Open content: 1-5 characters with numbers and letters

Queensland (Taken from https://www.ppq.com.au/):

1. AAA999 qld @@@###
2. AAA99 qld @@@##
3. 99AAA qld @@###
4. 999AAA qld @@@###
5. 9AAA9 qld @###@
6. 3 letters and 2 or 3 numbers (in any order) and 5 or 6 characters in total.
