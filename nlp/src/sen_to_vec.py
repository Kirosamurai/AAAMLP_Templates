import numpy as np

def sentence_to_vec(s, embedding_dict, stop_words, tokenizer):
    words = str(s).lower()
    words = tokenizer(words)

    words = [w for w in words if w not in stop_words]
    words = [w for w in words if w.isalpha()]

    M = []

    for w in words:
        if w in embedding_dict:
            M.append(embedding_dict[w])
    
    if len(M) == 0:
        return np.zeros(300)
    
    M = np.array(M)
    v = M.sum(axis=0)

    return v/np.sqrt((v**2).sum())
