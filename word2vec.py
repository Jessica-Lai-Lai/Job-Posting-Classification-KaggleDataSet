import numpy as np
from gensim.models import Word2Vec
from features import get_features_w2v
from classifier import search_C, train_model, eval_model


def search_hyperparams(Xt_train, y_train, Xt_val, y_val):
    """Search the best values of hyper-parameters for Word2Vec as well as the
    regularisation parameter C for logistic regression, using the validation set.

    Args:
        Xt_train, Xt_val (list(list(list(str)))): Lists of (tokenised) documents (each
            represented as a list of tokenised sentences where a sentence is a
            list of tokens) for training and validation, respectively.
        y_train, y_val: Dense vectors (np.ndarray) of class labels for training
            and validation, respectively. Each element of the vector is either
            0 or 1.

    Returns:
        dict(str : union(int, float)): The best values of hyper-parameters.
    """
    # TODO: tune at least two of the many hyper-parameters of Word2Vec 
    #       (e.g. vector_size, window, negative, alpha, epochs, etc.) as well as
    #       the regularisation parameter C for logistic regression
    #       using the validation set.

    # The code below needs to be modified.
    #best_params = dict()
    #best_C = 1.  # sklearn default
    #best_acc = 0.
    #best_params['C'] = best_C
    #assert 'C' in best_params
    #return best_params

    best_acc = 0.
    best_params = {}

    # Possible values for Word2Vec hyperparameters
    vector_sizes = [100, 200, 300]
    window_sizes = [3, 5, 7]
    #C_values = [0.01, 0.1, 1, 10, 100]  # Regularization parameter for logistic regression

    for vector_size in vector_sizes:
        for window in window_sizes:
            # Train Word2Vec model with current hyperparameters
            word_vectors = train_w2v(Xt_train, vector_size=vector_size, window=window)

            # Extract features using Word2Vec
            X_train_w2v = get_features_w2v(Xt_train, word_vectors)
            X_val_w2v = get_features_w2v(Xt_val, word_vectors)

            # Search for best C using validation set
            best_C = search_C(X_train_w2v, y_train, X_val_w2v, y_val)
            print(f'Best C found: {best_C} for vector_size={vector_size} and window={window}')

            # Evaluate model performance
            model = train_model(X_train_w2v, y_train, best_C)
            acc = eval_model(X_val_w2v, y_val, model)
            print(f'Accuracy: {acc:.4f}')

            if acc > best_acc:
                best_acc = acc
                best_params = {'C': best_C, 'vector_size': vector_size, 'window': window}

    return best_params



def train_w2v(Xt_train, vector_size=200, window=5, min_count=5, negative=10, epochs=3, seed=101, workers=10,
              compute_loss=False, **kwargs):
    """Train a Word2Vec model.

    Args:
        Xt_train (list(list(list(str)))): A list of (tokenised) documents (each
            represented as a list of tokenised sentences where a sentence is a
            list of tokens).
        See https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
        for descriptions of the other arguments.

    Returns:
        gensim.models.keyedvectors.KeyedVectors: A mapping from words (string) to their embeddings
            (np.ndarray)
    """
    #sentences_train = [sent for doc in Xt_train for sent in doc]
    # TODO: train the Word2Vec model
    #print(f'Training word2vec using {len(sentences_train):,d} sentences ...')
    # The code below needs to be modified.
    #w2v_model = None
    #return w2v_model.wv
    # check the input

    sentences_train = [sent for doc in Xt_train for sent in doc]

    print(f'Training word2vec using {len(sentences_train):,d} sentences ...')

    # Train Word2Vec model
    w2v_model = Word2Vec(sentences=sentences_train,
                         vector_size=vector_size,
                         window=window,
                         min_count=min_count,
                         negative=negative,
                         epochs=epochs,
                         seed=seed,
                         workers=workers,
                         compute_loss=compute_loss)

    return w2v_model.wv

