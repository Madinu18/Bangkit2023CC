import csv
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify

NUM_WORDS = 30000
EMBEDDING_DIM = 100
MAXLEN = 100
PADDING = 'post'
OOV_TOKEN = "<OOV>"
TRAIN_FILE = "./dataset/processed_train.csv"  # cloud storage url
STOPWORDS_FILE = "./dataset/stopwordbahasa.csv"  # cloud storage url
STOPWORDS = []


def load_stopwords(stopwords_file=STOPWORDS_FILE):
    with open(stopwords_file, 'r') as f:
        stopwords = []
        reader = csv.reader(f)
        for row in reader:
            stopwords.append(row[0])

        return stopwords


STOPWORDS = load_stopwords(STOPWORDS_FILE)


def remove_stopwords(sentence, stopwords=STOPWORDS):
    sentence = sentence.lower()

    words = sentence.split()
    no_words = [w for w in words if w not in stopwords]
    sentence = " ".join(no_words)

    return sentence


def parse_data_from_file(filename):
    sentences = []
    labels = []
    with open(filename, 'r', encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            sentence = row[6]
            sentence = remove_stopwords(sentence)
            sentences.append(sentence)

    return sentences


train_sentences = parse_data_from_file(TRAIN_FILE)


def fit_tokenizer(train_sentences, num_words, oov_token):
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(train_sentences)

    return tokenizer


tokenizer = fit_tokenizer(train_sentences, NUM_WORDS, OOV_TOKEN)


def seq_and_pad(sentences, tokenizer, padding, maxlen):
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, padding=padding, maxlen=maxlen)

    return padded_sequences


model = tf.keras.models.load_model('toxic_comment_model_var_1.h5')

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        text = request.form['sentence']
        if text is None:
            return jsonify({"error": "No text"})
        
        try:
            sentence = text
            sentence = remove_stopwords(sentence)
            sentences = []
            sentences.append(sentence)

            sentences_padded_seq = seq_and_pad(sentences, tokenizer, PADDING, MAXLEN)

            prediction = model.predict(sentences_padded_seq)
            data = {"prediction": float(prediction),
                    "text" : str(text)}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})
    
    return "OK"

if __name__ == "__main__":
    app.run(debug=True)
