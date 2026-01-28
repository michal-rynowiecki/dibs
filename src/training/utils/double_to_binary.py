from read_input import read_data
import json

if __name__ == "__main__":
    preds_path = '/Users/michal/Projects/sentiment/data/predictions/eng_laptop_preds_double.jsonl'
    data_path = '/Users/michal/Projects/sentiment/data/raw/subtask_3/eng/eng_laptop_train_alltasks.jsonl'
    output_path = '/Users/michal/Projects/sentiment/data/processed/eng_laptop_binary_input.jsonl'
    f = open(output_path, 'w')

    predicted_double = read_data(preds_path)
    data = read_data(data_path)

    ready_data = []
    for point in predicted_double:
        new_data = {}
        new_data['ID'] = point
        new_data['text'] = data[point]['Text']
        new_data['aspect'] = predicted_double[point]['aspect_predicted_tags']
        new_data['opinion'] = predicted_double[point]['opinion_predicted_tags']
        new_data['gold pairs'] = [(x['Aspect'], x['Opinion']) for x in data[point]['Quadruplet']]
        ready_data.append(new_data)
    
    for point in ready_data:
        json.dump(point, f)
        f.write("\n")