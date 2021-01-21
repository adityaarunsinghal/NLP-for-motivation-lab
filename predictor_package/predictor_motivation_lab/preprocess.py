import nltk
import pandas as pd
from collections import Counter
    
def pp_entry(entry):
    entry = entry.lower()
    pos_split = nltk.pos_tag(nltk.word_tokenize(entry))
    tags = [each[1] for each in pos_split]
    return(dict(Counter(tags)))

def pp_row(entry, scores):
    dct = pp_entry(entry)
    for i in range(1, len(scores)+1):
        exec("dct[f'SCORE_{i}']=scores[i-1]")
    return(dct)

def pp_set(entries, scores, coder):
    dct = {}
    for i in range(len(entries)):
        aug = pp_row(entries[i], scores[i])
        aug['SCORER'] = coder
        dct[i] = aug
    return(dct)

def pp_entries(entries_series):
    
    """
    input = Pandas Series of all Utterances
    output = Transcribed Dataframe readable by trainer/predictor
    """

    dcts = entries_series.apply(lambda x: pp_entry(x))
    return(pd.DataFrame(list(dcts)).fillna(0))

