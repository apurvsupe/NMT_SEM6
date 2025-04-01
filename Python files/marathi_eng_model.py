# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.callbacks import EarlyStopping

import tensorflow as tf
import json
# import tensorflow_text as tf_txt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, RepeatVector, Dropout, TimeDistributed
import sentencepiece as spm
import pathlib
import re
import random



# %%
AUTOTUNE = tf.data.AUTOTUNE


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# %%
class Config:
    INPUT_FILE_PATH : pathlib.PosixPath = pathlib.Path("/kaggle/input/parallel-corpus/Tatoeba.en-mr.mr")
    TARGET_FILE_PATH : pathlib.PosixPath = pathlib.Path("/kaggle/input/parallel-corpus/Tatoeba.en-mr.en")

    OPUS_IN_FILE_PATH : pathlib.PosixPath = pathlib.Path("/kaggle/input/opus-datasets/XLEnt.en-mr.mr")
    OPUS_OUT_FILE_PATH : pathlib.PosixPath = pathlib.Path("/kaggle/input/opus-datasets/XLEnt.en-mr.en")

    LOKMAT_IN_FILE_PATH : pathlib.PosixPath = pathlib.Path("/kaggle/input/lokmat-lifestyle/lokmat-lifestyle_train.mr")
    LOKMAT_OUT_FILE_PATH : pathlib.PosixPath = pathlib.Path("/kaggle/input/lokmat-lifestyle/lokmat-lifestyle_train.en")


    KAGGLE_FILE_PATH : pathlib.PosixPath = pathlib.Path("/kaggle/input/parallel-corpus/data.csv")

    CHECKPOINT_PATH : pathlib.PosixPath = pathlib.Path("/kaggle/working/model_29_03_v1.keras")

    INPUT_VOCAB_PATH : pathlib.PosixPath = pathlib.Path("/kaggle/working/inputVocab.json")
    TARGET_VOCAB_PATH : pathlib.PosixPath = pathlib.Path("/kaggle/working/targetVocab.json")

    PROCESSED_INPUT_PATH : pathlib.PosixPath = pathlib.Path("/kaggle/working/input_modified.txt")
    PROCESSED_TARGET_PATH : pathlib.PosixPath = pathlib.Path("/kaggle/working/target_modified.txt")

    SPM_INPUT_MODEL_PATH = pathlib.Path("/kaggle/working/mr_spm.model")
    SPM_TARGET_MODEL_PATH = pathlib.Path("/kaggle/working/en_spm.model")

    BATCH_SIZE : int = 64
    BUFFER_SIZE : int = 10000
    MAX_VOCAB_SIZE : int = 30000
    LSTM_UNITS : int = 256
    EMBEDDING_DIM : int = 256

    DROPOUT_RATE : float = 0.3
    LEARNING_RATE : float = 0.001
    EPOCHS : int = 20
    LABEL_SMOOTHING : float = 0.1


# %%
class Helper:
    def _init_(self):
        self.INPUT_TOKENIZER = None
        self.TARGET_TOKENIZER = None

        self.INPUT_VOCAB_SIZE = None
        self.TARGET_VOCAB_SIZE = None

        self.MAX_INPUT_LEN = None
        self.MAX_TARGET_LEN = None

        self.INPUT_VOCAB = None
        self.TARGET_VOCAB = None


        # These are the tensorflow datasets, for faster computations
        self.TRAIN_DATASET = None
        self.VAL_DATASET = None
        self.TEST_DATASET = None


    def save_state(self, file_path: str):
        state = {
            "INPUT_VOCAB_SIZE": self.INPUT_VOCAB_SIZE,
            "TARGET_VOCAB_SIZE": self.TARGET_VOCAB_SIZE,
            "MAX_INPUT_LEN": self.MAX_INPUT_LEN,
            "MAX_TARGET_LEN": self.MAX_TARGET_LEN
        }

        # Save tokenizers as file paths since SentencePiece models can't be directly serialized
        if self.INPUT_TOKENIZER and self.TARGET_TOKENIZER:
            state["INPUT_TOKENIZER_PATH"] = str(Config.SPM_INPUT_MODEL_PATH)
            state["TARGET_TOKENIZER_PATH"] = str(Config.SPM_TARGET_MODEL_PATH)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(state, file, indent=4)

    def load_state(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as file:
            state = json.load(file)

        self.INPUT_VOCAB_SIZE = state.get("INPUT_VOCAB_SIZE")
        self.TARGET_VOCAB_SIZE = state.get("TARGET_VOCAB_SIZE")
        self.MAX_INPUT_LEN = state.get("MAX_INPUT_LEN")
        self.MAX_TARGET_LEN = state.get("MAX_TARGET_LEN")

        # Load tokenizers if paths exist
        input_tokenizer_path = state.get("INPUT_TOKENIZER_PATH")
        target_tokenizer_path = state.get("TARGET_TOKENIZER_PATH")

        if input_tokenizer_path and target_tokenizer_path:
            import sentencepiece as spm
            self.INPUT_TOKENIZER = spm.SentencePieceProcessor()
            self.TARGET_TOKENIZER = spm.SentencePieceProcessor()
            self.INPUT_TOKENIZER.load(input_tokenizer_path)
            self.TARGET_TOKENIZER.load(target_tokenizer_path)

    
    @staticmethod
    def load_dataset(input_path: pathlib.PosixPath, target_path: pathlib.PosixPath = None):
            input_lines = []
            target_lines = []

            if input_path.suffix.lower() == ".csv":
                df = pd.read_csv(input_path)
                input_lines = df.iloc[:, 1].tolist()

            elif input_path.suffix.lower() == ".xlsx":
                df = pd.read_excel(input_path)
                input_lines = df.iloc[:, 1].tolist()

            else:
                input_text = input_path.read_text(encoding="utf-8")
                input_lines = [line for line in input_text.splitlines() if line.strip()]

            if target_path:
                if target_path.suffix.lower() == ".csv":
                    df = pd.read_csv(target_path)
                    target_lines = df.iloc[:, 0].tolist()

                elif target_path.suffix.lower() == ".xlsx":
                    df = pd.read_excel(target_path)
                    target_lines = df.iloc[:, 0].tolist()

                else:
                    target_text = target_path.read_text(encoding="utf-8")
                    target_lines = [line for line in target_text.splitlines() if line.strip()]

            if target_path and len(input_lines) != len(target_lines):
                print(f"WARNING: Input has {len(input_lines)} but target has {len(target_lines)} lines")

                min_length = min(len(input_lines), len(target_lines))
                input_lines = input_lines[:min_length]
                target_lines = target_lines[:min_length]

            return input_lines, target_lines if target_path else input_lines

    def normalize_text(self, text: str, is_marathi: bool=False) -> str:

            if is_marathi:
              text = re.sub(r"[^\u0900-\u097F.?!,\s]", " ", text)
            else:
              text = text.lower()
              text = re.sub(r"[^a-z.?!,\s]", " ", text)

            text = re.sub(r"\s+", " ", text).strip()

            return text


    def prepare_data(self, input_sentences, target_sentences):
            noramlized_inputs = [self.normalize_text(text, is_marathi=False) for text in input_sentences]
            noramlized_targets = [self.normalize_text(text, is_marathi=True) for text in target_sentences]

            return noramlized_inputs, noramlized_targets

    def train_spm(self, input_sentences, target_sentences):
            input_sentences, target_sentences = self.prepare_data(input_sentences, target_sentences)

            with open(Config.PROCESSED_INPUT_PATH, "w", encoding="utf-8") as file:
              file.write("\n".join(input_sentences))

            with open(Config.PROCESSED_TARGET_PATH, "w", encoding="utf-8") as file:
              file.write("\n".join(target_sentences))

            # Input Model
            spm.SentencePieceTrainer.train(
                    input=str(Config.PROCESSED_INPUT_PATH),
                    model_prefix=str(Config.SPM_INPUT_MODEL_PATH).replace(".model", ""),
                    vocab_size=Config.MAX_VOCAB_SIZE,
                    character_coverage=1.0,
                    model_type="unigram",
                    pad_id=0,
                    unk_id=1,
                    bos_id=2,
                    eos_id=3,
                    pad_piece="<PAD>",
                    unk_piece="<UNK>",
                    bos_piece="<SOS>",
                    eos_piece="<EOS>"
                )

            # Target Model
            spm.SentencePieceTrainer.train(
                    input=str(Config.PROCESSED_TARGET_PATH),
                    model_prefix=str(Config.SPM_TARGET_MODEL_PATH).replace(".model", ""),
                    vocab_size=Config.MAX_VOCAB_SIZE,
                    character_coverage=0.9995,  # Higher for Marathi to capture more characters
                    model_type="unigram",
                    pad_id=0,
                    unk_id=1,
                    bos_id=2,
                    eos_id=3,
                    pad_piece="<PAD>",
                    unk_piece="<UNK>",
                    bos_piece="<SOS>",
                    eos_piece="<EOS>"
                )

            self.INPUT_TOKENIZER = spm.SentencePieceProcessor()
            self.TARGET_TOKENIZER = spm.SentencePieceProcessor()

            self.INPUT_TOKENIZER.load(str(Config.SPM_INPUT_MODEL_PATH))
            self.TARGET_TOKENIZER.load(str(Config.SPM_TARGET_MODEL_PATH))

            self.INPUT_VOCAB_SIZE = self.INPUT_TOKENIZER.get_piece_size()
            self.TARGET_VOCAB_SIZE = self.TARGET_TOKENIZER.get_piece_size()

            print(f"Input vocabulary size: {self.INPUT_VOCAB_SIZE}")
            print(f"Target vocabulary size: {self.TARGET_VOCAB_SIZE}")

    def encode_data(self, input_texts, target_texts):
            input_ids = []

            for text in input_texts:
              ids = self.INPUT_TOKENIZER.encode(text, add_bos=False, add_eos=False)
              input_ids.append(ids)


            target_ids_input = []
            target_ids_output = []

            for text in target_texts:
                ids_input = [self.TARGET_TOKENIZER.bos_id()] + self.TARGET_TOKENIZER.encode(text, add_bos=False, add_eos=False)
                target_ids_input.append(ids_input)



                ids_output = self.TARGET_TOKENIZER.encode(text, add_bos=False, add_eos=False) + [self.TARGET_TOKENIZER.eos_id()]
                target_ids_output.append(ids_output)


            self.MAX_INPUT_LEN = max(len(ids) for ids in input_ids)
            self.MAX_TARGET_LEN = max(len(ids) for ids in target_ids_input)

            padded_input_ids = pad_sequences(input_ids, maxlen=self.MAX_INPUT_LEN, padding="post")
            padded_target_ids_input = pad_sequences(target_ids_input, maxlen=self.MAX_TARGET_LEN, padding="post")
            padded_target_ids_output = pad_sequences(target_ids_output, maxlen=self.MAX_TARGET_LEN, padding="post")


            return (padded_input_ids, padded_target_ids_input), padded_target_ids_output

    def create_tf_dataset(self, encoder_inputs, decoder_inputs, decoder_outputs, batch_size=Config.BATCH_SIZE, buffer_size=Config.BUFFER_SIZE, shuffle=True):

        dataset = tf.data.Dataset.from_tensor_slices(
            ((encoder_inputs, decoder_inputs), decoder_outputs)
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size)

        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.repeat(Config.EPOCHS).prefetch(buffer_size=Config.BUFFER_SIZE)


        return dataset

    def get_batches(self, train_data, val_data, test_data):
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data

        # Training the SentencePiece Model
        self.train_spm(X_train, y_train)

        # Encoding and padding the data
        (train_encoder_inputs, train_decoder_inputs), train_decoder_outputs = self.encode_data(X_train, y_train)
        (val_encoder_inputs, val_decoder_inputs), val_decoder_outputs = self.encode_data(X_val, y_val)
        (test_encoder_inputs, test_decoder_inputs), test_decoder_outputs = self.encode_data(X_test, y_test)

        # Creating Tensorflow datasets for optimization
        self.TRAIN_DATASET = self.create_tf_dataset(
            train_encoder_inputs, train_decoder_inputs, train_decoder_outputs,
            batch_size = Config.BATCH_SIZE
        )
        self.VAL_DATASET = self.create_tf_dataset(
            val_encoder_inputs, val_decoder_inputs, val_decoder_outputs,
            batch_size = Config.BATCH_SIZE, shuffle=False
        )
        self.TEST_DATASET = self.create_tf_dataset(
            test_encoder_inputs, test_decoder_inputs, test_decoder_outputs,
            batch_size = Config.BATCH_SIZE, shuffle=False
        )


        steps_per_epoch = len(train_decoder_inputs)//Config.BATCH_SIZE
        validation_steps = len(val_encoder_inputs)//Config.BATCH_SIZE

        return self.TRAIN_DATASET, self.VAL_DATASET, self.TEST_DATASET, steps_per_epoch, validation_steps

    def split_data(self, input_lines, target_lines, train_size=0.8, val_size=0.1, test_size=0.1):

        X_train, X_temp, y_train, y_temp = train_test_split(
            input_lines, target_lines, train_size=train_size, random_state=42
        )

        val_ratio = val_size / (val_size +  test_size)

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, train_size=val_ratio, random_state=42
        )

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# %%
@tf.keras.utils.register_keras_serializable()
class Encoder(tf.keras.Model):
  def __init__(self, vocab_size : int, embedding_dim : int, units : int, **kwargs):
    super(Encoder, self).__init__(**kwargs)

    self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
    self.lstm = tf.keras.layers.LSTM(units, return_state=True, return_sequences=True)

  @tf.function
  def call(self, inputs):
    x = self.embedding_layer(inputs)
    outputs, lstm_h, lstm_c = self.lstm(x)
    # outputs, forward_h, forward_c, backward_h, backward_c = self.lstm(x)
    
    # lstm_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
    # lstm_c = tf.keras.layers.Concatenate()([forward_c, backward_c])
    return outputs, lstm_h, lstm_c

  def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "units": self.units
        })
        return config

  @classmethod
  def from_config(cls, config):
        return cls(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            units=config["units"]
        )

  def build_from_config(self, config):
        pass

# %%
@tf.keras.utils.register_keras_serializable()
class Decoder(tf.keras.Model):
  def __init__(self, vocab_size : int, embedding_dim : int, units : int, **kwargs):
    super(Decoder, self).__init__(**kwargs)

    self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
    self.lstm = tf.keras.layers.LSTM(units, return_state=True, return_sequences=True)
    self.fc = tf.keras.layers.Dense(vocab_size, activation="softmax")

  @tf.function
  def call(self, inputs, hidden_state, cell_state):
    x = self.embedding_layer(inputs)
    outputs, state_h, state_c = self.lstm(x, initial_state=[hidden_state, cell_state])
    logits = self.fc(outputs)
    return logits, state_h, state_c

  def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "units": self.units
        })
        return config

  @classmethod
  def from_config(cls, config):
    return cls(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        units=config["units"]
    )

  def build_from_config(self, config):
      pass


# %%
@tf.keras.utils.register_keras_serializable()
class Seq2Seq(tf.keras.Model):
  def __init__(self,input_vocab_size, target_vocab_size, **kwargs):
    super(Seq2Seq, self).__init__(**kwargs)
    self.encoder = Encoder(input_vocab_size, Config.EMBEDDING_DIM, Config.LSTM_UNITS)
    self.decoder = Decoder(target_vocab_size, Config.EMBEDDING_DIM, Config.LSTM_UNITS)
    self.input_vocab_size = input_vocab_size
    self.target_vocab_size = target_vocab_size

  @tf.function
  def call(self, inputs):
    encoder_input, decoder_input = inputs
    encoder_outputs, state_h, state_c = self.encoder(encoder_input)
    decoder_outputs, _, _ = self.decoder(decoder_input, state_h, state_c)
    return decoder_outputs

  def get_config(self):
        config = super().get_config()
        config.update({
            "input_vocab_size": self.input_vocab_size,
            "target_vocab_size": self.target_vocab_size
        })
        return config

  @classmethod
  def from_config(cls, config):
        return cls(
            input_vocab_size=config["input_vocab_size"],
            target_vocab_size=config["target_vocab_size"]
        )

  def build_from_config(self, config):
        # The build method doesn't need to reinitialize the entire model
        # It should ensure the model is built with the correct input shapes
        # If needed, you can add specific build logic here
        pass

# %%
helper = Helper()
helper.load_state(r"C:\College\General\Code\sem_6\project\json file\helper_state_mr_en.json")

# %%
tf.config.run_functions_eagerly(True)
optimizer = tf.keras.optimizers.Adam(clipnorm=2.0)

# %%
def translate_sentence(sentence, encoder, decoder, helper):
    # Preprocess the input sentence
    raw = sentence
    sentence = helper.normalize_text(sentence, is_marathi=True)
    sentence = helper.INPUT_TOKENIZER.encode(sentence)  # FIXED

    sentence = pad_sequences([sentence], maxlen=helper.MAX_INPUT_LEN, padding="post")  # FIXED

    # Encode the input sentence
    encoder_outputs, state_h, state_c = encoder(sentence)

    # Prepare the decoder input (<SOS>)
    target_seq = np.array([[helper.TARGET_TOKENIZER.piece_to_id("<SOS>")]])  # FIXED

    decoded_sentence = []

    for _ in range(helper.MAX_TARGET_LEN):
        predictions, state_h, state_c = decoder(target_seq, state_h, state_c)


        predicted_id = np.argmax(predictions[0, -1, :])


        if predicted_id == helper.TARGET_TOKENIZER.piece_to_id("<EOS>"):
            break

        word = helper.TARGET_TOKENIZER.id_to_piece(int(predicted_id))

        # print(f"Raw token: {word}")
        # Remove '_' at the beginning of the word (handling SentencePiece's unigram encoding)
        cleaned_word = word.replace("▁", "")
        decoded_sentence.append(cleaned_word)

        # decoded_sentence.append(helper.TARGET_TOKENIZER.id_to_piece(int(predicted_id)))

        # Update target sequence
        target_seq = np.array([[predicted_id]])

    decoded_sentence = " ".join(decoded_sentence)
    print(f"Sentence : {raw} ==> Translation: {decoded_sentence}")
    
    return decoded_sentence

# %% [markdown]
# ### ***FOLLOW THE STEPS TO LOAD MODEL (Mentioned in code below)***

# %%
model1 = tf.keras.models.load_model(r"C:\College\General\Code\sem_6\project\marathi_model\mr_en_model_v1.keras", custom_objects={"Encoder" : Encoder, "Decoder": Decoder, "Seq2Seq" : Seq2Seq})

# %%
translate_sentence("मी तुझ्यावर प्रेम करतो", model1.encoder, model1.decoder, helper)
model1.load_weights(r"C:\College\General\Code\sem_6\project\marathi_model\mr_en_model_weights.weights.h5")
model1.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# %%
translate_sentence("मी तुझ्यावर प्रेम करतो", model1.encoder, model1.decoder, helper)

