from sklearn.feature_extraction import DictVectorizer

data = [
    {"price": 123111, "rooms": 4, "neighbourhood": "San Lorenzo"},
    {"price": 37, "rooms": 2, "neighbourhood": "Trastevere"},
    {"price": 73, "rooms": 3, "neighbourhood": "Pigneto"},
]
# But we need to have the neighbourhood repreented as a number for ML
# We could:
{"San Lorenzo": 1, "Trastevere": 2, "Pigneto": 3}
# Not a good idea, it might suggest San Lorenzo is lower than Pigneto,
# Or maybe that San Lorenzo + Trastevere = Pigneto

# Instead we can use 1-hot encoding
# So each value gets 1 bit, or alternatively we get a dict saying:

{"price": 123111, "rooms": 4, "San Lorenzo": 1, "Trastevere": 0, "Pigneto": 0}

# We can use sklearn to make this a whole lot easier:
vec = DictVectorizer(sparse=False, dtype=int)
# spare means theres not much data in the matrix, ie most of the data is 0
# If a matrix is sparse we can store the data more efficiently
one_hot_data = vec.fit_transform(data)
print(one_hot_data)
# We appear to have lost the column names, so we can use this func:

names = vec.get_feature_names_out()
print(names)
