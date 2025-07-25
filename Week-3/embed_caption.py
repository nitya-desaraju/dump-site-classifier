import re, string
import pickle
from cogworks_data.language import get_data_path
import gensim
from gensim.models.keyedvectors import KeyedVectors
from collections import defaultdict
import numpy as np
import math


punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))


def embed_caption(captions):

    #takes each caption from all captions and processes it
    def process_caption(captions):
        
        processed_cap = {}

        for caption in captions:
            processed_cap[caption.name] = punc_regex.sub('', caption.caption.lower())

        return processed_cap #{ID1: "caption1", ID2: "caption2", ID3: "caption3"}
    
    #tokenizes all information from all captions
    #input: array of caption strings
    def tokenize_words(captions):
        vocab = set()
        
        for caption in captions.values():
            tokens = caption.split()
            vocab.update(tokens)
        
        # {word_string: glove_vector, word_string2: glove_vector2, etc}
        W = {}

        path = get_data_path("glove.6B.200d.txt.w2v")
        glove = KeyedVectors.load_word2vec_format(path, binary=False)

        for vocab_word in vocab:
            if vocab_word in glove:
                W[vocab_word] = glove[vocab_word]
            else:
                print(f"Error: word '{vocab_word}' not found in glove")
        
        return W
    
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

        with open('idf.pkl', 'wb') as f:
            pickle.dump(idf, f)

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


    #process the captions
    processed_captions = process_caption(captions)

    #get tokenized words and IDFs
    IDFs = calculate_idf(processed_captions)
    W = tokenize_words(processed_captions)

    #tokenize the captions
    #W_norm={caption_ID_1: W_norm_1,...}
    W_norm = tokenize_caption(W, IDFs, processed_captions)
    
    with open("W_norm.pkl", "wb") as f:
        pickle.dump(W_norm, f)

    return W_norm




    

    
