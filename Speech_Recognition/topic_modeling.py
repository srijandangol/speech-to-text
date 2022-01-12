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
import _thread

# with open('Python_Apps/Speech_Recognition/2021-05-15 15-03-00.txt') as f:
with open('Python_Apps/Speech_Recognition/textfile.txt') as f:
    lines = f.readlines()[0]


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
# print(data_clean)

cv = CountVectorizer(stop_words=['english','aah','aaah']) # Add more words
data_cv = cv.fit_transform(data_clean.transcript)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index

# EDA
data = data_dtm.transpose()
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
    print(topics)

# Show the plot for one user
_thread.start_new_thread(noun_print,())

plt.plot(polarity_transcript[0])
plt.title("USER")
plt.show()