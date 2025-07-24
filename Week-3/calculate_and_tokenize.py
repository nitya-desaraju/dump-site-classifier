import math
from collections import defaultdict
import numpy as np
import pickle


def calculate_idf(captions):
    """
    captions: List of preprocessed captions from process_caption()
    returns: dict mapping word -> IDF score
    """

    N = len(captions)
    doc_freq = defaultdict(int)

    for cap in captions:
        words = set(cap.split())
        for word in words:
            doc_freq[word] += 1

    idf = {word: math.log(N / (1 + freq)) for word, freq in doc_freq.items()}
    return idf


def tokenize_caption(W_words, idfs, captions):
    """
    W_words: embedding vectors for each word from tokenize_words()
    idfs: IDF scores for each word from calculate_idf()
    captions: processed caption text from process_caption()
    returns: dict mapping caption id -> normalized vector and saves to pkl
    """
    
    W_norms = {}

    for caption_id, caption_text in captions.items():
        caption_vec = np.zeros(200) #stores weighted vector
        words = caption_text.split() 

        for word in words:
            embedding = W_words[word]
            idf = idfs[word]
            caption_vec += embedding * idf #calculates vector
        
        magnitude = np.linalg.norm(caption_vec) #
        if magnitude > 0:
            norm = caption_vec / magnitude #normalize
        else:
            norm = caption_vec

        W_norms[caption_id] = norm

    #save
    with open('captions.pkl', 'rb') as f_in:
        captions_list = pickle.load(f_in)
    
    caption_map = {c.name: c for c in captions_list}

    for caption_id, w_norm_value in W_norms.items():
        if caption_id in caption_map:
            caption_map[caption_id].W_norm = w_norm_value

    with open('captions.pkl', 'wb') as f_out:
        pickle.dump(captions_list, f_out)

    return W_norms