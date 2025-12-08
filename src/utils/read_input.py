import json

def read_data(path: str):
    file = open(path, 'r')

    data = {}
    for item in file.readlines():
        my_dict = json.loads(item)
        line_id = my_dict.pop('ID')
        data[line_id] = my_dict

    return data