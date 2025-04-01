import socket
# from mr_model import translate_marathi_to_english  # Existing Marathi-to-English model
from eng_model import translate_sentence_eng  # English-to-Marathi model
from mr_model import translate_sentence_mr
from tensorflow.keras.preprocessing.sequence import pad_sequences
from eng_marthi_model import Encoder, Decoder, Seq2Seq, Helper
import numpy as np
import tensorflow as tf


# Model paths
# ENCODER_PATH = r"C:\College\General\Code\sem_6\project\models\encoder_model_v1.keras"
# DECODER_PATH = r"C:\College\General\Code\sem_6\project\models\decoder_model_v1.keras"
# INPUT_TOKENIZER_PATH = r"C:\College\General\Code\sem_6\project\models\ip_tokenizer.model"
# TARGET_TOKENIZER_PATH = r"C:\College\General\Code\sem_6\project\models\trg_tokenizer.model"
# MAX_LENGTH = 50
FLAG = False
MODEL = None
# Server configuration
HOST = "0.0.0.0"
PORT = 5000

# paths
weight_path = r'C:\College\General\Code\sem_6\project\models\en_mr_model_weights.weights.h5'
model_path = r'C:\College\General\Code\sem_6\project\models\en_mr_model_v2.keras'
helper = Helper()
helper.load_state(r"C:\College\General\Code\sem_6\project\json file\helper_state.json")
optimizer = tf.keras.optimizers.Adam(clipnorm=2.0)


# Create a socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)

print(f"Server started. Listening on {HOST}:{PORT}...")

optimizer = tf.keras.optimizers.Adam(clipnorm=2.0)
while True:

    client_socket, addr = server_socket.accept()
    print(f"Connection from {addr} established.")

    try:
        # Receive data from client
        data = client_socket.recv(1024).decode("utf-8")
        if not data:
            client_socket.close()
            continue

        # Determine translation direction
        if data.startswith("MR-EN|"):  # Marathi to English
            text = data.replace("MR-EN|", "").strip()
            translated_text = translate_sentence_mr(text, optimizer) or "Error: Marathi-to-English translation failed."

        elif data.startswith("EN-MR|"):  # English to Marathi
            text = data.replace("EN-MR|", "").strip()
            print(text)
            try:
                translated_text = translate_sentence_eng(text, optimizer)
                print("Here : \n",translated_text)
                if translated_text is None:
                    translated_text = "Error: English-to-Marathi translation failed."
            except Exception as e:
                translated_text = f"Error: Translation failed - {str(e)}"

        else:
            translated_text = "Error: Invalid request format."

        # Send translated text
        client_socket.send(translated_text.encode("utf-8"))

    except Exception as e:
        print(f"Error processing request: {e}")
        client_socket.send(b"Error: Server encountered an issue.")

    finally:
        client_socket.close()
