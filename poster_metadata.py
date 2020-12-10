'''
python file used to create the get the metadata into a csv from each of 
the .txt files in `data/Movie_Poster_Metadata` folder
'''
import pandas as pd
import os
import json
import re

def make_json_readable(s):
    '''
    method to change raw string from textfile json-compatible

    Parameters
    ==========
    `s`:
        complete string from text file

    Returns
    ==========
    updated json-formatted object with a comma delimiter and all values
    encapsulated by double quotes
    '''

    # replace new line-separated json-like objects with commma-separated objects
    s = s.replace("}\n{", "},{")

    # remove the rest of the new line characters
    s = s.replace("\n", "")

    # use regexp to find all appearances of this value corresponding to the _"id" key
    o = re.findall("ObjectId\(\"........................\"\)", s)
    
    for returns in o:
        # replace each appearance of this badly-formatted ObjectId("{}") value to
        # just the idea contained within the ObjectId parameter
        s = s.replace(returns, "\"" + returns[10:-2] + "\"")
    return s

def main():
    '''
    read in each textfile that contain the json-objects, create a DataFrame from 
    those objects and save this DataFrame into a csv file
    '''
    path = "data/Movie_Poster_Metadata/groundtruth"

    destinations = [os.path.join(path, file) for file in os.listdir(path)]
    json_objects = []

    for dest in destinations:
        # some of these files have utf-8 encoding and others have
        # utf-16 encoding, so I just haced a solution using an unrecommended try/except
        try:
            with open(dest, encoding="utf-8") as f:
                jsons_string = f.read()
        except Exception:
            with open(dest, encoding="utf-16") as f:
                jsons_string = f.read()
        s = make_json_readable(jsons_string)
        # call json loads function on the properly formatted string
        dictionary_list = json.loads('[%s]'%s)
        
        json_objects.extend(dictionary_list)
    
    # create a DataFrame from this lsit of dictionaries (json objects) and save to csv
    pd.DataFrame(json_objects).to_csv("data/movies-metadata.csv")


if __name__ == "__main__":
    main()