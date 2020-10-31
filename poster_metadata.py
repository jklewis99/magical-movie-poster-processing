import pandas as pd
import os
import json
import re

def make_json_readable(s):
    s= s.replace("}\n{", "},{")
    s = s.replace("\n", "")
    o = re.findall("ObjectId\(\"........................\"\)", s)
    for returns in o:
        s = s.replace(returns, "\"" + returns[10:-2] + "\"")
    return s

def main():
    path = "Movie_Poster_Metadata/groundtruth"

    destinations = [os.path.join(path, file) for file in os.listdir(path)]
    json_objects = []

    for dest in destinations:
        try:
            with open(dest, encoding="utf-8") as f:
                jsons_string = f.read()
            s = make_json_readable(jsons_string)
            a = json.loads('[%s]'%s)
            json_objects.extend(a)
        except Exception:
            with open(dest, encoding="utf-16") as f:
                jsons_string = f.read()
            s = make_json_readable(jsons_string)
            a = json.loads('[%s]'%s)
            json_objects.extend(a)
    
    pd.DataFrame(json_objects).to_csv("posters-metadata.csv")


if __name__ == "__main__":
    main()