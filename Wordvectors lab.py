#!/usr/bin/env python
# coding: utf-8

# Part 1: Genism

# In this lab, I'll be working with "Sophie's World," a philosophical book that explores the history of philosophy. This book holds a special place in my heart. It's one of my all-time favorite books. I initially read it in French as a teenager and have now decided to revisit it, this time in English. I'm curious to explore some word vector analysis on its text.

# In[1]:


import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import random

from gensim.models import Word2Vec

from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE


# In[2]:


nltk.download('punkt')


# In[3]:


# read a file you have stored locally
# I added the Hunger Games for simplicity
file = open("Sophies world.txt", 'r', encoding='utf-8').read()

# first, remove unwanted new line and tab characters from the text
for char in ["\n", "\r", "\d", "\t"]:
   file = file.replace(char, " ")

# check
print(file[:100])


# In[4]:


# this is simplified for demonstration
def sample_clean_text(text: str):
    # step 1: tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)

    # step 2: tokenize each sentence into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # step 3: convert each word to lowercase
    tokenized_text = [[word.lower() for word in sent] for sent in tokenized_sentences]
    
    # return your tokens
    return tokenized_text

# call the function
tokens = sample_clean_text(text = file)

# check
print(tokens[:10])


# In[5]:


model = Word2Vec(tokens,vector_size=100)


# In[6]:


model.wv.key_to_index


# In[7]:


model.wv.get_vector("god", norm=True)


# In[8]:


model.wv.most_similar('god')


# In[9]:


model.wv.most_similar('love')


# In[10]:


#model.wv.similarity('mind', 'idea')


# In[11]:


#model.wv.similarity('world', 'question')


# In[12]:


model.wv.similarity('philosopher', 'question')


# In[13]:


import warnings

# Ignore the specific t-SNE warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._t_sne")

def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


x_vals, y_vals, labels = reduce_dimensions(model)
warnings.filterwarnings("default")


# In[14]:


def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=True):
    from plotly.offline import init_notebook_mode, iplot, plot
    import plotly.graph_objs as go

    trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
    data = [trace]

    if plot_in_notebook:
        init_notebook_mode(connected=True)
        iplot(data, filename='word-embedding-plot')
    else:
        plot(data, filename='word-embedding-plot.html')


def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))

try:
    get_ipython()
except Exception:
    plot_function = plot_with_matplotlib
else:
    plot_function = plot_with_plotly

plot_function(x_vals, y_vals, labels)


# Visualizing a Smaller Sample of 300 Words

# In[15]:


import warnings

# Ignore the warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._t_sne")

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def reduce_dimensions(model, num_words_to_display=100):
    # Extract the words and their vectors
    words = list(model.wv.index_to_key)
    vectors = [model.wv[word] for word in words]

    # Reduce dimensions using t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    vectors_2D = tsne.fit_transform(vectors[:num_words_to_display])

    return vectors_2D, words

# Define the number of words you want to display
num_words_to_display = 300

# Replace 'model' with your loaded word vector model
vectors_2D, words = reduce_dimensions(model, num_words_to_display)

# Plot the word vectors
plt.figure(figsize=(12, 12))
for i, label in enumerate(words[:num_words_to_display]):
    x, y = vectors_2D[i]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points')

plt.show()
warnings.filterwarnings("default")


# In the visualizations, we notice that some words stick together. For instance, words like "god," "spirit," and "angel" are huddled, hinting at something spiritual. Then, names like "Alberto," "Hilde," and "Sophie" are bunched up, pointing to key characters. Words like "philosopher" and "question" are cozy, showing a connection between deep thinking and inquiry. Also, words like "mother," "father," and "home" gather, suggesting a family connection. 

# Part 2: GloVe

# In[16]:


import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# In[17]:


embeddings_dict = {}


# In[ ]:


with open("glove.6B.100d.txt", 'r', encoding="utf-8") as f:
  for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector


# In[ ]:


def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))


# In[ ]:


lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding)


# In[ ]:


find_closest_embeddings(embeddings_dict["philosopher"])


# In[ ]:


find_closest_embeddings(embeddings_dict["philosopher"])[:5] 

                                    


# In[ ]:


print(find_closest_embeddings(embeddings_dict["love"])[:20])


# In[ ]:


print(find_closest_embeddings(embeddings_dict["love"] + embeddings_dict["life"])[:20])


# In[ ]:


print(find_closest_embeddings(embeddings_dict["love"] - embeddings_dict["hate"] + embeddings_dict["life"])[:20])


# I appreciate how the code above returns words like 'mother' and 'father' as top results. This outcome is logically sound, as it aligns with the transformed context generated by subtracting 'hate' and adding 'life' to the word 'love' in the word embeddings.

# In[ ]:


print(find_closest_embeddings(embeddings_dict["human"] + embeddings_dict["peace"])[:20])


# In[ ]:


words =  list(embeddings_dict.keys())
vectors = [embeddings_dict[word] for word in words]
X = np.asarray(vectors)


# In[ ]:


import warnings
# Ignore the warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._t_sne")

tsne = TSNE(n_components=2, random_state=0)
Y = tsne.fit_transform(X[:1000])
warnings.filterwarnings("default")


# In[ ]:


import warnings
# Ignore the warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._t_sne")
tsne = TSNE(n_components=2, random_state=0)
words =  list(embeddings_dict.keys())
vectors = [embeddings_dict[word] for word in words]
Y = tsne.fit_transform(vectors[:1000])
plt.scatter(Y[:, 0], Y[:, 1])

for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
plt.show()
warnings.filterwarnings("default")


# In[ ]:


import warnings
# Ignore the warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._t_sne")

tsne = TSNE(n_components=2, random_state=0, perplexity=30, learning_rate=200.0)
words =  list(embeddings_dict.keys())
vectors = [embeddings_dict[word] for word in words]
Y = tsne.fit_transform(vectors[:250])
plt.scatter(Y[:, 0], Y[:, 1])

for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
plt.show()
warnings.filterwarnings("default")


# In[ ]:


import warnings
# Ignore the warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._t_sne")

tsne = TSNE(n_components=2, random_state=0, perplexity=30, learning_rate=200.0)
words =  list(embeddings_dict.keys())
vectors = [embeddings_dict[word] for word in words]
Y = tsne.fit_transform(vectors[:100])
plt.scatter(Y[:, 0], Y[:, 1])

for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
plt.show()
warnings.filterwarnings("default")


# In[ ]:


import warnings
# Ignore the warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._t_sne")

tsne = TSNE(n_components=2, random_state=0, perplexity=30, learning_rate=200.0)
words =  list(embeddings_dict.keys())
vectors = [embeddings_dict[word] for word in words]
Y = tsne.fit_transform(vectors[:80])
plt.scatter(Y[:, 0], Y[:, 1])

for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
plt.show()
warnings.filterwarnings("default")


# When I first looked for words most similar to "love" in the book "Sophie's World" using the Gensim model, the results were a bit surprising. It gave words like "moon," "contain," "marx," and "eggs," which don't seem immediately connected to "love."
# 
# In contrast, when I used the GloVe model to search for "love," the results were much more intuitive. It provided words like "passion," "dream," "true," "life," and "friends," which are words that we naturally associate with the concept of "love."  
