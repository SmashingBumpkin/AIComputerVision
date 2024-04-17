import nltk

# nltk.download("omw-1.4")
# nltk.download("tagsets")
# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("wordnet")

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# word tokenizes based on words
# sent tokenizes based on splitting a sentence into smaller sentences

example_string = "a quick brown fox jumps over the lazy dog. This is a second sentence. THis is a sentence, it goes on after the comma"


#########TOKENIZATION
sent_res = sent_tokenize(
    example_string
)  # sentence tokenizer (splits sentence to list of sentences)

word_res = word_tokenize(
    example_string
)  # word tokenizer (splits words to list of words)


########REMOVING UNNECESSARY WORDS
# List of words that may be less useful for sentence meaning... apparently according to someone
stop_words = set(stopwords.words("english"))


filtered_list = []

# removes "stop_words" from the sentence
for word in word_res:
    if word.casefold() not in stop_words:  # casefold ignores string case
        filtered_list.append(word)


############GETTING WORD STEMS
# Gets the route words from a word (eg "sees" comes from "see" as in, "to see")
stemmer = SnowballStemmer(language="english")

quote = "THe USS Discovery's discoveries were discussed at length for the discoveries that resulted from the Discovery's discoveries"
# quote = "Pears, 4 peppers, 3 cloves, 4 cloves of garlic"
quote_tokenized = word_tokenize(quote)

stem_words = [stemmer.stem(word) for word in quote_tokenized]

# print(stem_words)

# overstemming is if 2 unrelated words are stemmed to the same root
# understemming is if a word is not stemmed properly


#########TAGGING
tags = nltk.pos_tag(quote_tokenized)  # "Parts of speech tag"
# print(tags)
# Each tag means something, eg NN is a noun, a proper noun (name) is NNP, verbs are VBx etc


#########Lemmatization - gets the route of a word of some sort
lemmatizer = WordNetLemmatizer()
res = lemmatizer.lemmatize("worst", pos="a")
print(res)
