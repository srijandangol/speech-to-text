import tkinter as tk
from tkinter import *
import argparse
import os
import ast
import queue
import sounddevice
import vosk
import sys
import _thread
from datetime import datetime
import re
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text

import pandas as pd

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import numpy as np
import math
from gensim import matutils, models
import scipy.sparse
from nltk import word_tokenize, pos_tag

# ui
is_running=True
final_text  =[]
root = Tk()
fls = StringVar()
fls2 = StringVar()
fls3 = StringVar()
fls_diff = StringVar()

##
canvas = Canvas(root, width=800, height=650)
canvas.create_text(100,20,fill="darkblue",font="Times 20 italic bold",
                        text="Click the bubbles that are multiples of two.")
##

fls.set('Start Record') # Update
fls2.set('Your Speech')
fls3.set('')


wrapper = LabelFrame(root,text='Spech to Text')
wrapper.pack(fill="both",expand="yes",padx=10,pady=10)

lbl3 = Label(wrapper,textvariable=fls3)
lbl3.pack()

lbl = Label(wrapper,textvariable=fls)
lbl.pack()

lbl2 = Label(wrapper,textvariable=fls2)
lbl2.pack()

def record():
    global is_running
    is_running=True
    btn1.pack_forget()
    btn2.pack(padx=20)
    fls.set('Recording') # Update
    fls3.set('') # Update
    _thread.start_new_thread(startrecord,())

def stop():
    global is_running,final_text
    is_running = False
    fls.set('Start Recording') 
    btn1.pack(padx=20)
    btn2.pack_forget()
    topic_modeling(final_text)

def startrecord():
    # recorder
    q = queue.Queue()
    def int_or_str(text):
        """Helper function for argument parsing."""
        try:
            return int(text)
        except ValueError:
            return text

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-l', '--list-devices', action='store_true',
        help='show list of audio devices and exit')

    args, remaining = parser.parse_known_args()

    if args.list_devices:
        print(sounddevice.query_devices())
        parser.exit(0)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])

    parser.add_argument(
        '-f', '--filename', type=str, metavar='FILENAME',
        help='audio file to store recording to')

    parser.add_argument(
        '-m', '--model', type=str, metavar='MODEL_PATH',
        help='Path to the model')

    parser.add_argument(
        '-d', '--device', type=int_or_str,
        help='input device (numeric ID or substring)')

    parser.add_argument(
        '-r', '--samplerate', type=int, help='sampling rate')
    args = parser.parse_args(remaining)
    try:
        if args.model is None:
            args.model = "Python_Apps/Speech_Recognition/model"
        if not os.path.exists(args.model):
            print ("Please download a model for your language from https://alphacephei.com/vosk/models")
            print ("and unpack as 'model' in the current folder.")
            parser.exit(0)
        if args.samplerate is None:
            device_info = sounddevice.query_devices(args.device, 'input')
            # soundfile expects an int, sounddevice provides a float:
            args.samplerate = int(device_info['default_samplerate'])

        model = vosk.Model(args.model)

        if args.filename:
            dump_fn = open(args.filename, "wb")
        else:
            dump_fn = None

        with sounddevice.RawInputStream(samplerate=args.samplerate, blocksize = 8000, device=args.device, dtype='int16',
                                channels=1, callback=callback):
                print('#' * 80)
                print('Press Ctrl+C to stop the recording')
                print('#' * 80)

                rec = vosk.KaldiRecognizer(model, args.samplerate)
                while is_running:
                    data = q.get()
                    if rec.AcceptWaveform(data):
                        fls2.set('Listening..')
                        text = ast.literal_eval(rec.FinalResult())['text']
                        global final_text
                        final_text.append(text)
                        new_text = ' '.join(final_text)
                        new_text = [new_text[i:i+110] for i in range(0, len(new_text), 110)]
                        fls3.set('\n'.join(new_text))
                        # print(new_text)
                        # pass
                    else:
                        # print(rec.PartialResult())
                        text = ast.literal_eval(rec.PartialResult())['partial']
                        text =  [text[i:i+110] for i in range(0, len(text), 110)]
                        fls2.set('\n'.join(text))
                    if dump_fn is not None:
                        print('helooooooooooooooooo')
                        dump_fn.write(data)
                else:
                    fls2.set('')
                    parser.exit(0)

    except KeyboardInterrupt:
        print('\nDone')
        parser.exit(0)
    except Exception as e:
        parser.exit(type(e).__name__ + ': ' + str(e))

def topic_modeling(final_text):
    lines = final_text[0]

    def clean_text_round1(text):
        '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\w*\d\w*', '', text)
        # get rid of more things....
        text = re.sub('[‘’“”…]', '', text)
        text = re.sub('\n', '', text)
        return text

    lines = clean_text_round1(lines)

    data_combined = {'user': [lines]}
    # print(data_combined)
    data_clean = pd.DataFrame.from_dict(data_combined).transpose()
    data_clean.columns = ['transcript']
    data_clean = data_clean.sort_index()
    print('data_clean',data_clean)

    cv = CountVectorizer(stop_words=['english','aah','aaah']) # Add more words
    data_cv = cv.fit_transform(data_clean.transcript)
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_dtm.index = data_clean.index

    # EDA
    data = data_dtm.transpose()
    # Find the top 20 words said by each comedian
    top_dict = {}
    for c in data.columns:
        top = data[c].sort_values(ascending=False).head(20)
        top_dict[c]= list(zip(top.index, top.values))

    wc = WordCloud(stopwords={'and','are','you','your'}, background_color="white", colormap="Dark2", #add stopwords
                max_font_size=150, random_state=42)
    wc.generate(data_clean.transcript['user'])
        
    # plt.subplot(1, 1,1)
    # plt.imshow(wc, interpolation="bilinear")
    # plt.axis("off")
    # plt.show()


    # sentiment
    def pol(x):
        # print(x) # each word
        return TextBlob(x).sentiment.polarity

    sub = lambda x: TextBlob(x).sentiment.subjectivity

    data_clean['polarity'] = data_clean['transcript'].apply(pol)
    data_clean['subjectivity'] = data_clean['transcript'].apply(sub)

    def split_text(text, n=10):
        '''Takes in a string of text and splits into n equal parts, with a default of 10 equal parts.'''

        # Calculate length of text, the size of each chunk of text and the starting points of each chunk of text
        length = len(text)
        size = math.floor(length / n)
        start = np.arange(0, length, size)
        
        # Pull out equally sized pieces of text and put it into a list
        split_list = []
        for piece in range(n):
            split_list.append(text[start[piece]:start[piece]+size])
        return split_list

    # Let's create a list to hold all of the pieces of text
    list_pieces = []
    for t in data_clean.transcript:
        split = split_text(t)
        list_pieces.append(split)

    # Calculate the polarity for each piece of text

    polarity_transcript = []
    for lp in list_pieces:
        polarity_piece = []
        for p in lp:
            polarity_piece.append(TextBlob(p).sentiment.polarity)
        polarity_transcript.append(polarity_piece)

    def noun_print():
        # Topic Modeling
        # Let's create a function to pull out nouns from a string of text
        def nouns(text):
            '''Given a string of text, tokenize the text and pull out only the nouns.'''
            is_noun = lambda pos: pos[:2] == 'NN'
            tokenized = word_tokenize(text)
            all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)] 
            return ' '.join(all_nouns)

        data_nouns = pd.DataFrame(data_clean.transcript.apply(nouns))

        # Re-add the additional stop words since we are recreating the document-term matrix
        add_stop_words = ['like', 'im', 'know', 'just', 'dont', 'thats', 'right', 'people',
                        'youre', 'got', 'gonna', 'time', 'think', 'yeah', 'said','ah']
        stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

        # Recreate a document-term matrix with only nouns
        cvn = CountVectorizer(stop_words=stop_words)
        data_cvn = cvn.fit_transform(data_nouns.transcript)
        data_dtmn = pd.DataFrame(data_cvn.toarray(), columns=cvn.get_feature_names())
        data_dtmn.index = data_nouns.index

        # Create the gensim corpus
        corpusn = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmn.transpose()))

        # Create the vocabulary dictionary
        id2wordn = dict((v, k) for k, v in cvn.vocabulary_.items())

        # Create the gensim corpus
        corpusn = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmn.transpose()))

        # Create the vocabulary dictionary
        id2wordn = dict((v, k) for k, v in cvn.vocabulary_.items())

        # Let's start with 2 topics
        ldan = models.LdaModel(corpus=corpusn, num_topics=2, id2word=id2wordn, passes=10)

        # Let's try topics = 3
        ldan = models.LdaModel(corpus=corpusn, num_topics=3, id2word=id2wordn, passes=10)

        # Let's try 4 topics
        ldan = models.LdaModel(corpus=corpusn, num_topics=4, id2word=id2wordn, passes=10)

        # Our final LDA model (for now)
        ldan = models.LdaModel(corpus=corpusn, num_topics=4, id2word=id2wordn, passes=80)

        topics = []
        topic = ldan.print_topics()
        for top1 in topic:
            for top2 in top1[1].split(' + '):
                topics.append(top2[7:-1])
        topics = list(set(topics))
        # print(topics)
        fls3.set(f'You are talking about\n{" ".join(topics)}')

    # Show the plot for one user
    _thread.start_new_thread(noun_print,())

    # plt.plot(polarity_transcript[0])
    # plt.title("USER")
    # plt.show()
# btn3 = Button(wrapper,text="Exit",command=lambda:exit())
# btn3.pack(padx=1)

btn1 = Button(wrapper,text="Record",command=record)
btn1.pack(padx=20)

btn2 = Button(wrapper,text="Stop",command=stop)


root.title('Speech to Text.')
root.geometry("720x600")
root.resizable(False,False)
root.mainloop()