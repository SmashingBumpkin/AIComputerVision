from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

# word tokenizes based on words
# sent tokenizes based on splitting a sentence into smaller sentences

example_string = "a quick brown fox jumps over the lazy dog. This is a second sentence. THis is a sentence, it goes on after the comma"

sent_res = sent_tokenize(example_string)
print(sent_res)

word_res = word_tokenize(example_string)
print(word_res)
