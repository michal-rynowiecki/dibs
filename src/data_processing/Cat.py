import json
from collections import Counter

import json

def split_json_categories(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:
        
        cat1_counter = Counter()
        cat2_counter = Counter()    
        for line in f_in:                
            # Parse the original JSON line
            data = json.loads(line)
            
            # Update each quadruplet in the list
            for quad in data.get("Quadruplet", []):
                category = quad.get("Category", "")
            
                cat1, cat2 = category.split("#", 1)
                # Add new keys
                quad["Cat1"] = cat1
                quad["Cat2"] = cat2

                cat1_counter[cat1] += 1
                cat2_counter[cat2] += 1
                # Remove the old key
                del quad["Category"]
        
            # Write the modified dictionary back to the new file
            f_out.write(json.dumps(data) + '\n')
    # Display results
    print("### Cat1 (Entity) Counts ###")
    for cat, count in cat1_counter.items():
        print(f"{cat}: {count}")

    print("\n### Cat2 (Attribute) Counts ###")
    for cat, count in cat2_counter.items():
        print(f"{cat}: {count}")

# --- Execute ---
split_json_categories('/Users/michal/Projects/sentiment/data/raw/subtask_3/eng/eng_laptop_train_alltasks.jsonl', '/Users/michal/Projects/sentiment/data/processed/categories_eng_laptop_train_alltasks.jsonl')