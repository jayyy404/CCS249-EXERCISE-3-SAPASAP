

import re
import nltk
import wikipedia
import math
from collections import defaultdict
from nltk.util import ngrams
from nltk.tokenize import word_tokenize


nltk.download('punkt')


page = wikipedia.page("Natural language processing")
corpus = page.content[:1000]

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) 
    tokens = word_tokenize(text)
    return tokens


tokens = preprocess_text(corpus)

# Function to train n-gram model with Laplace smoothing
def train_ngram_model(tokens, n):
    model = defaultdict(lambda: defaultdict(lambda: 0))
    vocab = set(tokens)

    ngram_list = list(ngrams(tokens, n, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))

    for ngram in ngram_list:
        prefix, word = tuple(ngram[:-1]), ngram[-1]
        model[prefix][word] += 1


    for prefix in model:
        total_count = float(sum(model[prefix].values())) + len(vocab) 
        for word in model[prefix]:
            model[prefix][word] = (model[prefix][word] + 1) / total_count  

    return model, vocab


bigram_model, vocab_bigram = train_ngram_model(tokens, 2)
trigram_model, vocab_trigram = train_ngram_model(tokens, 3)

# Function to calculate perplexity with Laplace smoothing using log probabilities
def calculate_perplexity(model, sentence, n, vocab_size):
    tokens = preprocess_text(sentence)
    ngram_list = list(ngrams(tokens, n, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))

    log_probability = 0
    N = len(ngram_list)

    for ngram in ngram_list:
        prefix, word = tuple(ngram[:-1]), ngram[-1]
        prob = (model[prefix].get(word, 1)) / (sum(model[prefix].values()) + vocab_size)  
        log_probability += math.log(prob)  

    perplexity = math.exp(-log_probability / N) 
    return round(perplexity, 2)

#J.K Rowling Quote
test_sentence = "If you want to see the true nature of a man, watch how he treats his inferiors, not his equals."
text_token = preprocess_text(test_sentence)
print ("Tokens of Test Sentence:", text_token)

# Compute perplexity with Laplace smoothing
bigram_perplexity = calculate_perplexity(bigram_model, test_sentence, 2, len(vocab_bigram))
trigram_perplexity = calculate_perplexity(trigram_model, test_sentence, 3, len(vocab_trigram))


print(f"Bigram model perplexity -> Test Sentence \"{test_sentence}\" -> Score: {bigram_perplexity}")
print(f"Trigram model perplexity -> Test Sentence \"{test_sentence}\" -> Score: {trigram_perplexity}")
