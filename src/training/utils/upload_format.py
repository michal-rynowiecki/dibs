import json

path = '/Users/michal/Projects/sentiment/data/predictions/test/eng_restaurant_preds_va_test.jsonl'
out_path = '/Users/michal/Projects/sentiment/data/predictions/upload/upload_restaurant.jsonl'
f = open(path, 'r')

data_raw = []
for i in f.readlines():
    data_raw.append(json.loads(i))
f.close()
def combine_data(data_list):
    grouped_data = {}

    for item in data_list:
        item_id = item['id']
        
        # Initialize entry if ID is seen for the first time
        if item_id not in grouped_data:
            grouped_data[item_id] = {
                "ID": item_id,
                #"text": item['text'],
                "Quadruplet": []
            }
        
        # Construct the quadruplet element
        quad = {
            "Aspect": item['aspect'],
            "Category": f"{item['cat1']}#{item['cat2']}",
            "Opinion": item['opinion'],
            "VA": f"{item['valence']:.2f}#{item['arousal']:.2f}"
        }
        
        grouped_data[item_id]["Quadruplet"].append(quad)
    
    return list(grouped_data.values())

# Process the data
result = combine_data(data_raw)

f_out = open(out_path, 'w')
# Print as JSON for visualization
for res in result:
    json.dump(res, f_out)
    f_out.write('\n')