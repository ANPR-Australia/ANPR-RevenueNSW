# ANPR-RevenueNSW
Exploring ANPR solutions for number plates in Australia

You need to compile openalpr from source, and know the location of the source files for the pip install below.

```
python3 -m venv alpr
source alpr/bin/activate

python -m pip install --editable <$location_of_openalpr_source_code/src/bindings/python>
python -m pip install --editable .   

```

For the purposes of pattern matching, the alpha-numeric formats mentioned below exist for New South Wales standard plates. These do not include any custom plates and are taken from https://www.myplates.com.au/

1. AA-99-AA
2. AAA-99A
3. AAA-999
4. AA-999
5. AA-9999
6. N-AA99A
7. N-AA999
10. 999-AAA
11. 99-AAAA
12. 99-AAA
