import json

path = '/home/michal/Documents/research/sentiment/DimABSA2026/task-dataset/track_a/subtask_1/eng/eng_laptop_train_alltasks.jsonl'

def create_BIO_tags(path):

    file = open(path, 'r')

    for line in file.readlines():
        read = json.loads(line)

        text = read['Text'].split()
        flattened = [i['Aspect'].split() for i in read['Quadruplet']]
        #aspect = [phrase.split() for phrase in read['Aspect']]

        print(text)
        print(flattened)

        BIO2 = ['O'] * len(text)
        '''
        for sub_aspect in aspect:
            len_aspect = len(sub_aspect)

            # Slide over text to find matching spans (subtract the length of the aspect because the aspect can't start if there isn't enough words in the text to match with the words in the aspect)
            for i in range(len(text) - len_aspect + 1):
                if text[i:i+len_aspect] == sub_aspect:
                    # If it isn't marked by a previous aspect
                    if BIO2[i]=='O':
                        # First word -> B, rest -> I
                        BIO2[i] = 'B'
                        for j in range(1, len_aspect):
                            BIO2[i+j] = 'I'
        
        # Print result
        for w, label in zip(text, BIO2):
            print(w, label)

        print('\n')
        '''
create_BIO_tags(path)