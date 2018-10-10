"""
Model definition.
"""
import json

from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, TimeDistributed, Conv1D, GlobalMaxPooling1D, Activation
from keras.layers.merge import Concatenate
from keras.models import Model, model_from_json
from keras.losses import categorical_crossentropy

from anago.layers import CRF


def save_model(model, weights_file, params_file):
    with open(params_file, 'w') as f:
        params = model.to_json()
        json.dump(json.loads(params), f, sort_keys=True, indent=4)
        model.save_weights(weights_file)


def load_model(weights_file, params_file):
    with open(params_file) as f:
        model = model_from_json(f.read(), custom_objects={'CRF': CRF})
        model.load_weights(weights_file)

    return model


class BiLSTMCRF(object):
    """A Keras implementation of BiLSTM-CRF for sequence labeling.

    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360
    """

    def __init__(self,
                 num_labels,
                 word_vocab_size,
                 char_vocab_size=None,
                 word_embedding_dim=100,
                 char_embedding_dim=25,
                 word_lstm_size=100,
                 char_lstm_size=25,
                 fc_dim=100,
                 dropout=0.5,
                 embeddings=None,
                 use_char=True,
                 use_crf=True):
        """Build a Bi-LSTM CRF model.

        Args:
            word_vocab_size (int): word vocabulary size.
            char_vocab_size (int): character vocabulary size.
            num_labels (int): number of entity labels.
            word_embedding_dim (int): word embedding dimensions.
            char_embedding_dim (int): character embedding dimensions.
            word_lstm_size (int): character LSTM feature extractor output dimensions.
            char_lstm_size (int): word tagger LSTM output dimensions.
            fc_dim (int): output fully-connected layer size.
            dropout (float): dropout rate.
            embeddings (numpy array): word embedding matrix.
            use_char (boolean): add char feature.
            use_crf (boolean): use crf as last layer.
        """
        super(BiLSTMCRF).__init__()
        self._char_embedding_dim = char_embedding_dim
        self._word_embedding_dim = word_embedding_dim
        self._char_lstm_size = char_lstm_size
        self._word_lstm_size = word_lstm_size
        self._char_vocab_size = char_vocab_size
        self._word_vocab_size = word_vocab_size
        self._fc_dim = fc_dim
        self._dropout = dropout
        self._use_char = use_char
        self._use_crf = use_crf
        self._embeddings = embeddings
        self._num_labels = num_labels

    def build(self):
        # build word embedding
        word_ids = Input(batch_shape=(None, None), dtype='int32', name='word_input')
        inputs = [word_ids]
        if self._embeddings is None:
            word_embeddings = Embedding(input_dim=self._word_vocab_size,
                                        output_dim=self._word_embedding_dim,
                                        mask_zero=True,
                                        name='word_embedding')(word_ids)
        else:
            word_embeddings = Embedding(input_dim=self._embeddings.shape[0],
                                        output_dim=self._embeddings.shape[1],
                                        mask_zero=True,
                                        weights=[self._embeddings],
                                        name='word_embedding')(word_ids)

        # build character based word embedding
        if self._use_char:
            char_ids = Input(batch_shape=(None, None, None), dtype='int32', name='char_input')
            inputs.append(char_ids)
            char_embeddings = Embedding(input_dim=self._char_vocab_size,
                                        output_dim=self._char_embedding_dim,
                                        mask_zero=True,
                                        name='char_embedding')(char_ids)
            char_embeddings = TimeDistributed(Bidirectional(LSTM(self._char_lstm_size)))(char_embeddings)
            word_embeddings = Concatenate()([word_embeddings, char_embeddings])

        word_embeddings = Dropout(self._dropout)(word_embeddings)
        z = Bidirectional(LSTM(units=self._word_lstm_size, return_sequences=True))(word_embeddings)
        z = Dense(self._fc_dim, activation='tanh')(z)

        if self._use_crf:
            crf = CRF(self._num_labels, sparse_target=False)
            loss = crf.loss_function
            pred = crf(z)
        else:
            loss = 'categorical_crossentropy'
            pred = Dense(self._num_labels, activation='softmax')(z)

        model = Model(inputs=inputs, outputs=pred)

        return model, loss


class ELModel(object):
    """
    A Keras implementation of ELMo BiLSTM-CRF for sequence labeling.
    """

    def __init__(self,
                 num_labels,
                 word_vocab_size,
                 char_vocab_size=None,
                 word_embedding_dim=100,
                 char_embedding_dim=25,
                 word_lstm_size=100,
                 char_lstm_size=25,
                 fc_dim=100,
                 dropout=0.5,
                 embeddings=None):
        """Build a Bi-LSTM CRF model.

        Args:
            word_vocab_size (int): word vocabulary size.
            char_vocab_size (int): character vocabulary size.
            num_labels (int): number of entity labels.
            word_embedding_dim (int): word embedding dimensions.
            char_embedding_dim (int): character embedding dimensions.
            word_lstm_size (int): character LSTM feature extractor output dimensions.
            char_lstm_size (int): word tagger LSTM output dimensions.
            fc_dim (int): output fully-connected layer size.
            dropout (float): dropout rate.
            embeddings (numpy array): word embedding matrix.
        """
        self._char_embedding_dim = char_embedding_dim
        self._word_embedding_dim = word_embedding_dim
        self._char_lstm_size = char_lstm_size
        self._word_lstm_size = word_lstm_size
        self._char_vocab_size = char_vocab_size
        self._word_vocab_size = word_vocab_size
        self._fc_dim = fc_dim
        self._dropout = dropout
        self._embeddings = embeddings
        self._num_labels = num_labels

    def build(self):
        # build word embedding
        word_ids = Input(batch_shape=(None, None), dtype='int32', name='word_input')
        if self._embeddings is None:
            word_embeddings = Embedding(input_dim=self._word_vocab_size,
                                        output_dim=self._word_embedding_dim,
                                        mask_zero=True,
                                        name='word_embedding')(word_ids)
        else:
            word_embeddings = Embedding(input_dim=self._embeddings.shape[0],
                                        output_dim=self._embeddings.shape[1],
                                        mask_zero=True,
                                        weights=[self._embeddings],
                                        name='word_embedding')(word_ids)

        # build character based word embedding
        char_ids = Input(batch_shape=(None, None, None), dtype='int32', name='char_input')
        char_embeddings = Embedding(input_dim=self._char_vocab_size,
                                    output_dim=self._char_embedding_dim,
                                    mask_zero=True,
                                    name='char_embedding')(char_ids)
        char_embeddings = TimeDistributed(Bidirectional(LSTM(self._char_lstm_size)))(char_embeddings)

        elmo_embeddings = Input(shape=(None, 1024), dtype='float32')

        word_embeddings = Concatenate()([word_embeddings, char_embeddings, elmo_embeddings])

        word_embeddings = Dropout(self._dropout)(word_embeddings)
        z = Bidirectional(LSTM(units=self._word_lstm_size, return_sequences=True))(word_embeddings)
        z = Dense(self._fc_dim, activation='tanh')(z)

        crf = CRF(self._num_labels, sparse_target=False)
        loss = crf.loss_function
        pred = crf(z)

        model = Model(inputs=[word_ids, char_ids, elmo_embeddings], outputs=pred)

        return model, loss


    
class ELModelClassif(object):
    """
    A Keras implementation of ELMo BiLSTM-CRF for sequence labeling.
    """

    def __init__(self,
                 num_labels,
                 word_vocab_size,
                 char_vocab_size=None,
                 word_embedding_dim=100,
                 char_embedding_dim=25,
        #         word_lstm_size=100,
                 char_lstm_size=25,
                 filters = 250,
                 kernel_size = 3,
                 fc_dim=100,
                 dropout=0.5,
                 dense_activation='softmax',
                 embeddings=None):
        """Build a Bi-LSTM CRF model.

        Args:
            word_vocab_size (int): word vocabulary size.
            char_vocab_size (int): character vocabulary size.
            num_labels (int): number of entity labels.
            word_embedding_dim (int): word embedding dimensions.
            char_embedding_dim (int): character embedding dimensions.
            word_lstm_size (int): character LSTM feature extractor output dimensions.
            char_lstm_size (int): word tagger LSTM output dimensions.
            fc_dim (int): output fully-connected layer size.
            dropout (float): dropout rate.
            embeddings (numpy array): word embedding matrix.
        """
        self._char_embedding_dim = char_embedding_dim
        self._word_embedding_dim = word_embedding_dim
        self._char_lstm_size = char_lstm_size
       # self._word_lstm_size = word_lstm_size
        self._char_vocab_size = char_vocab_size
        self._word_vocab_size = word_vocab_size
        self._fc_dim = fc_dim
        self._dropout = dropout
        self._embeddings = embeddings
        self._num_labels = num_labels
        self._filters = filters
        self._kernel_size = kernel_size
        self._dense_activation = dense_activation

    def build(self):
        # build word embedding
        word_ids = Input(batch_shape=(None, None), dtype='int32', name='word_input')
        if self._embeddings is None:
            word_embeddings = Embedding(input_dim=self._word_vocab_size,
                                        output_dim=self._word_embedding_dim,
                                        mask_zero=False,
                                        name='word_embedding')(word_ids)
        else:
            word_embeddings = Embedding(input_dim=self._embeddings.shape[0],
                                        output_dim=self._embeddings.shape[1],
                                        mask_zero=False,
                                        weights=[self._embeddings],
                                        name='word_embedding')(word_ids)

        # build character based word embedding
        char_ids = Input(batch_shape=(None, None, None), dtype='int32', name='char_input')
        char_embeddings = Embedding(input_dim=self._char_vocab_size,
                                    output_dim=self._char_embedding_dim,
                                    mask_zero=False,
                                    name='char_embedding')(char_ids)
        char_embeddings = TimeDistributed(Bidirectional(LSTM(self._char_lstm_size)))(char_embeddings)

        elmo_embeddings = Input(shape=(None, 1024), dtype='float32')

        word_embeddings = Concatenate()([word_embeddings, char_embeddings, elmo_embeddings])

        word_embeddings = Dropout(self._dropout)(word_embeddings)
        
        mod = Conv1D(self._filters, self._kernel_size, padding='same', activation ='relu', strides = 1)(word_embeddings)
        mod = GlobalMaxPooling1D()(mod)

        preds = []
        
        for l in self._num_labels:    
        
            submod = Dense(self._fc_dim)(mod)
            submod = Dropout(0.5)(submod)
            submod = Activation('relu')(submod)
            subpreds = Dense(l, activation =self._dense_activation)(submod)
            preds.append(subpreds)
        
        loss = categorical_crossentropy

        model = Model(inputs=[word_ids, char_ids, elmo_embeddings], outputs=preds)

        return model, loss