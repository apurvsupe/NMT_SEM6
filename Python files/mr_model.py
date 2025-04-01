import re
import numpy as np
import tensorflow as tf
import sentencepiece as spm
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from marathi_eng_model import Encoder, Decoder, Seq2Seq, Helper

def translate_sentence_mr_eng(sentence, encoder, decoder, helper):
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


def translate_sentence_mr(sentence, optimizer):
    # print("1\n")
    helper = Helper()
    helper.load_state(r"C:\College\General\Code\sem_6\project\json file\helper_state_mr_en.json")
    # print(helper.MAX_TARGET_LEN)
    model_path = r"C:\College\General\Code\sem_6\project\marathi_model\mr_en_model_v1.keras"
    weight_path = r"C:\College\General\Code\sem_6\project\marathi_model\mr_en_model_weights.weights.h5"
    # optimizer = tf.keras.optimizers.Adam(clipnorm=2.0)
    # print("Optimized\n")
    model1 = tf.keras.models.load_model(model_path, custom_objects={"Encoder" : Encoder, "Decoder": Decoder, "Seq2Seq" : Seq2Seq})
    model1.load_weights(weight_path)
    translate_sentence_mr_eng("I", model1.encoder, model1.decoder, helper)
    model1.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model1.load_weights(weight_path)

    sent = translate_sentence_mr_eng(sentence, model1.encoder, model1.decoder, helper)
    return sent