import json
from random import random, shuffle

if __name__ == "__main__":
    path = '/Users/michal/Projects/sentiment/data/raw/subtask_3/eng/eng_laptop_train_alltasks.jsonl'
    output_path = '/Users/michal/Projects/sentiment/data/processed/bin_laptop_train_alltasks.jsonl'
    f = open(path, 'r')

    data = []
    for line in f.readlines():
        # Initialize empty lists (with null as an available combo value everywhere)
        aspects = {'NULL'}
        opinions = {'NULL'}
        temp = json.loads(line)

        # Retrieve all the aspects and opinions
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
    f.close()

    f_out = open(output_path, 'w')  
    shuffle(data)
    for i in data:
        json.dump(i, f_out)
        f_out.write('\n')