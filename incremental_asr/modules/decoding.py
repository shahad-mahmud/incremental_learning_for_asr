import torch
from itertools import groupby

def greedy_decode(logits, tokenizer):
    arg_max = torch.argmax(logits, dim=2)
    
    batch_outs = []
    for tokens in arg_max:
        tokens = tokens.tolist()
        tokens = [t for t in tokens if t != 0]
        decoded_tokens = tokenizer.decode_ids(tokens)
        filtered = filter_output(decoded_tokens)
        text = ''.join(filtered)
        batch_outs.append(text)
        print(text)
    return batch_outs
        
def filter_output(string_pred: list, blank_id=0):
    string_out = [
        v
        for i, v in enumerate(string_pred)
        if i == 0 or v != string_pred[i - 1]
    ]
    string_out = [i[0] for i in groupby(string_out)]
    string_out = list(filter(lambda elem: elem != blank_id, string_out))
    return string_out  