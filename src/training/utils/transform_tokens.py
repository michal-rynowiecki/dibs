def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word_idx = None
    
    for word_idx in word_ids:
        if word_idx is None:
            # Special tokens ([CLS], [SEP])
            new_labels.append(0)
        elif word_idx != current_word_idx:
            # This is the FIRST subword of a new word
            new_labels.append(labels[word_idx])
        else:
            new_labels.append(0)
            # Option B: Or use the same label (if your task requires it)
            #new_labels.append(labels[word_idx])
            
        current_word_idx = word_idx
    
    return new_labels



def get_entities(tokenizer, input_ids, predictions, labels):
    words = tokenizer.convert_ids_to_tokens(input_ids)

    def retrieve_tokens(predictions):
        entities = []
        current_entity_tokens = []
        
        for word, pred in zip(words, predictions):
            is_subword = word.startswith("##") or word.startswith("#")
            clean_word = word.lstrip("#")

            if pred == 1:
                # If we were already building an entity, save it before starting a new one
                if current_entity_tokens:
                    entities.append(" ".join(current_entity_tokens))
                    current_entity_tokens = []
                current_entity_tokens.append(clean_word)
                
            elif pred == 2:
                if is_subword and current_entity_tokens:
                    current_entity_tokens[-1] += clean_word
                else:
                    current_entity_tokens.append(clean_word)
                    
            elif pred == 0:
                if is_subword and current_entity_tokens:
                    # Even if pred is 0, if it's a subword of a labeled word, we attach it
                    current_entity_tokens[-1] += clean_word
                else:
                    if current_entity_tokens:
                        entities.append(" ".join(current_entity_tokens))
                        current_entity_tokens = []
        
        return entities
    
    entities = retrieve_tokens(predictions)
    entities_gold = retrieve_tokens(labels)

    return entities, entities_gold 


def get_entities_batch(tokenizer, input_ids_batch, predictions_batch, labels_batch):
    """
    Retrieves entities for a batch of inputs.
    
    Args:
        tokenizer: The tokenizer used for encoding.
        input_ids_batch: Tensor of shape [batch_size, seq_len]
        predictions_batch: Tensor of shape [batch_size, seq_len]
        labels_batch: Tensor of shape [batch_size, seq_len]
        
    Returns:
        batch_entities: List of list of predicted entity strings per sample.
        batch_entities_gold: List of list of actual entity strings per sample.
    """
    
    # Helper to convert tensor to list if necessary
    def to_list(tensor_or_list):
        if hasattr(tensor_or_list, "tolist"):
            return tensor_or_list.tolist()
        return tensor_or_list

    # Ensure inputs are lists for easy iteration
    input_ids_batch = to_list(input_ids_batch)
    predictions_batch = to_list(predictions_batch)
    labels_batch = to_list(labels_batch)

    batch_entities = []
    batch_entities_gold = []

    # Iterate over the batch
    for i in range(len(input_ids_batch)):
        input_ids = input_ids_batch[i]
        preds = predictions_batch[i]
        labs = labels_batch[i]

        # Convert IDs to tokens for this specific sequence
        # Skip special tokens usually handled here, but raw conversion is fine 
        # based on your original snippet.
        words = tokenizer.convert_ids_to_tokens(input_ids)

        def retrieve_tokens(sequence_predictions):
            entities = []
            current_entity_tokens = []
            
            for word, pred in zip(words, sequence_predictions):
                # Handle subword prefixes for different tokenizers (BERT uses ##, others might use different schemes)
                is_subword = word.startswith("##") 
                clean_word = word.replace("##", "")
                
                # Skip [CLS], [SEP], [PAD] if necessary, or rely on pred=0
                if word in tokenizer.all_special_tokens:
                    continue

                if pred == 1:  # B-tag (Begin)
                    # If we were already building an entity, save it
                    if current_entity_tokens:
                        entities.append(" ".join(current_entity_tokens))
                        current_entity_tokens = []
                    current_entity_tokens.append(clean_word)
                    
                elif pred == 2:  # I-tag (Inside)
                    if is_subword and current_entity_tokens:
                        current_entity_tokens[-1] += clean_word
                    else:
                        current_entity_tokens.append(clean_word)
                        
                elif pred == 0:  # O-tag (Outside)
                    if is_subword and current_entity_tokens:
                        # Logic from original: subword of entity gets attached even if O
                        current_entity_tokens[-1] += clean_word
                    else:
                        if current_entity_tokens:
                            entities.append(" ".join(current_entity_tokens))
                            current_entity_tokens = []
            
            # Catch trailing entity at end of sequence
            if current_entity_tokens:
                entities.append(" ".join(current_entity_tokens))
            
            return entities
        
        # Process current sample
        sample_entities = retrieve_tokens(preds)
        sample_gold = retrieve_tokens(labs)
        
        batch_entities.append(sample_entities)
        batch_entities_gold.append(sample_gold)

    return batch_entities, batch_entities_gold