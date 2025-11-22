import json

file = open('/home/michal/Documents/research/sentiment/DimABSA2026/task-dataset/track_a/subtask_1/eng/eng_laptop_dev_task1.jsonl', 'r')

text2 = "I like the graphics , the fast graphics startup" 
aspect2 = ["graphics", "graphics startup"]

text_split2 = text2.split()
print("text split: ", text_split2)

aspect_split2 = [a.split() for a in aspect2]
print("aspect split: ", aspect_split2)


BIO2 = ['O'] * len(text_split2)

for sub_aspect in aspect_split2:
    len_aspect = len(sub_aspect)

    # Slide over text to find matching spans (subtract the length of the aspect because the aspect can't start if there isn't enough words in the text to match with the words in the aspect)
    for i in range(len(text_split2) - len_aspect + 1):
        if text_split2[i:i+len_aspect] == sub_aspect:
            # First word -> B, rest -> I
            BIO2[i] = 'B'
            for j in range(1, len_aspect):
                BIO2[i+j] = 'I'

for line in file.readlines():
    read = json.loads(line)

    text = read['Text'].split()
    aspect = [phrase.split() for phrase in read['Aspect']]

    BIO2 = ['O'] * len(text)

    print(text)
    print(aspect)

    for sub_aspect in aspect:
        len_aspect = len(aspect)

        # Slide over text to find matching spans (subtract the length of the aspect because the aspect can't start if there isn't enough words in the text to match with the words in the aspect)
        for i in range(len(text) - len_aspect + 1):
            if text[i:i+len_aspect] == sub_aspect:
                # First word -> B, rest -> I
                BIO2[i] = 'B'
                for j in range(1, len_aspect):
                    BIO2[i+j] = 'I'
    
    # Print result
    for w, label in zip(text, BIO2):
        print(w, label)
    
