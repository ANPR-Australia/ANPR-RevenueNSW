import sys
from openalpr import Alpr
import json

if __name__ == "__main__":
    openalpr_runtime = "/Users/sara/work/openalpr/runtime_data"
    alpr = Alpr("au", sys.argv[1], openalpr_runtime)
    if not alpr.is_loaded():
        print("Error loading OpenALPR")
        sys.exit(1)

    res = alpr.recognize_file(sys.argv[2])
    print(json.dumps(res, indent=4)) #for debugging only
    alpr.unload()
    
