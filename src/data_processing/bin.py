import json
from random import random, shuffle

if __name__ == "__main__":
    inference = True

    #path = '/Users/michal/Projects/sentiment/data/raw/subtask_3/eng/eng_restaurant_train_alltasks.jsonl'
    path = '/Users/michal/Projects/sentiment/data/predictions/test/eng_restaurant_preds_double_test.jsonl'
    output_path = '/Users/michal/Projects/sentiment/data/processed/bin_restaurant_test_alltasks.jsonl'
    f = open(path, 'r')

    data = []
    for line in f.readlines():
        # Initialize empty lists (with null as an available combo value everywhere)
        aspects = {'NULL'}
        opinions = {'NULL'}

        temp = json.loads(line)

        # Retrieve all the aspects and opinions
        if inference == False:
            for quad in temp['Quadruplet']:
                    aspects.add(quad['Aspect'])
                    opinions.add(quad['Opinion'])
            
            for aspect in aspects:
                for opinion in opinions:
                    flag = 0
                    for quad in temp['Quadruplet']:
                        if aspect == quad['Aspect'] and opinion == quad['Opinion']:
                            flag = 1
                    
                    data_point = {'ID': temp['ID'], 'Text': temp['Text'], 'Aspect': aspect, 'Opinion': opinion, 'exists': flag}
                        
                    r = random()
                    if not flag and aspect == 'NULL' and opinion == 'NULL' and r > 0.4:
                        continue
                    elif not flag and (aspect == 'NULL' or opinion == 'NULL') and r > 0.5:
                        continue
                    elif not flag and r > 0.6:
                        continue
                    data.append(data_point)
        else:
            aspects = set(temp['aspect_predicted_tags'])
            opinions = set(temp['opinion_predicted_tags'])

            for aspect in aspects:
                for opinion in opinions:
                    data_point = {'ID': temp['ID'], 'Text': temp['sentence'], 'Aspect': aspect, 'Opinion': opinion}
                    data.append(data_point)

    f.close()

    f_out = open(output_path, 'w')  
    
    if inference==False:
        shuffle(data)
    
    for i in data:
        json.dump(i, f_out)
        f_out.write('\n')