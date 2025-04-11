import nltk
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer


tokeniser = nltk.tokenize.TreebankWordTokenizer()
stopwords = frozenset(nltk.corpus.stopwords.words("english"))
trans_table = str.maketrans(dict.fromkeys(string.punctuation))


def tokenise_text(str_):
    """Tokenize a string of text.

    Args:
        str_: The input string of text.

    Returns:
        list(str): A list of tokens.
    """
    # for simplicity, remove non-ASCII characters
    str_ = str_.encode(encoding='ascii', errors='ignore').decode()
    return [t for t in tokeniser.tokenize(str_.lower().translate(trans_table)) if t not in stopwords]




def get_features_tfidf(Xr_fit, Xr_pred=None):
    """Compute the TF-IDF features of input documents.

    Args:
        Xr_fit (iterable(str)): Training documents as strings.
        Xr_pred (iterable(str)): Optional. Test documents as strings.

    Returns:
        X_fit: Sparse matrix of TF-IDF features for the training set.
        X_pred: Optional. Sparse matrix of TF-IDF features for the test set if provided.
    """

    # TODO: compute the TF-IDF features of the input documents.
    #   You may want to use TfidfVectorizer in the scikit-learn package,
    #   see https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    print('Generating features (TF-IDF) ...')

    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        analyzer=tokenise_text,  # Use the custom tokenizer
        max_features=5000,  # You can adjust the number of features (optional)
        #ngram_range=(1, 2)  # You can choose the n-gram range (optional)
    )

    # Fit the vectorizer on the training data
    X_fit = vectorizer.fit_transform(Xr_fit)

    # If there is a prediction set, transform it
    if Xr_pred is not None:
        X_pred = vectorizer.transform(Xr_pred)
        return X_fit, X_pred

    return X_fit




def document_to_vector(tokenised_doc, word_vectors):
    """Takes a (tokenised) document and turns it into a vector by aggregating
    its word vectors.

    Args:
        tokenised_doc (list(list(str))): A document represented as list of
            sentences. Each sentence is a list of tokens.
        word_vectors (gensim.models.keyedvectors.KeyedVectors): A mapping 
            from words (string) to their embeddings (np.ndarray)

    Returns:
        np.array: The aggregated word vector representing the input document.
    """
    # check the input
    assert isinstance(word_vectors, KeyedVectors)
    vector_size = word_vectors.vector_size


    #TODO: convert each document into a vector
    # Code below needs to be modified.
    #vec = None
    #return vec
    # check the input


    # Initialize a vector of zeros to accumulate word vectors
    vec = np.zeros(vector_size)

    # Count of valid words to average later
    count = 0

    # Iterate over the tokens in the document
    for sentence in tokenised_doc:
        for token in sentence:
            if token in word_vectors:  # Check if the token exists in the word vectors
                vec += word_vectors[token]  # Aggregate the word vectors
                count += 1  # Increment the valid word count

    # Return the average vector if count > 0, otherwise return a zero vector
    return vec / count if count > 0 else vec


def get_features_w2v(Xt, word_vectors):
    """Given a dataset of (tokenised) documents (each represented as a list of
    tokenised sentences), return a (dense) matrix of aggregated word vector for
    each document in the dataset.

    Args:
        Xt (list(list(list(str)))): A list of (tokenised) documents (each
            represented as a list of tokenised sentences where a sentence is a
            list of tokens).
        word_vectors (gensim.models.keyedvectors.KeyedVectors): A mapping 
            from words (string) to their embeddings (np.ndarray)

    Returns:
        np.ndarray: A matrix of features. The i-th row vector represents the i-th
            document in `Xr`.
    """
    print('Generating features (word2vec) ...')
    return np.vstack([document_to_vector(xt, word_vectors) for xt in tqdm(Xt)])

