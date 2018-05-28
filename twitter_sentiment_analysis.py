import nltk
nltk.data.path.append(
    '/media/light/UbuntuDrive/Python_Code/Propython/NLTK_HOME')

from nltk.corpus import twitter_samples
import random
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.collocations import *
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


twitter_samples.fileids()

strings = twitter_samples.strings('negative_tweets.json')

labeled_neg = [(x, 'neg') for x in strings]

pos_strings = twitter_samples.strings('positive_tweets.json')

labeled_pos = [(x, 'pos') for x in pos_strings]

labeled_pos

full_label_data = labeled_neg + labeled_pos

random.shuffle(full_label_data)

full_label_data

documents = [x for x, y in full_label_data]

train_data = documents[:7000]
train_label = labels[:7000]

test_data = documents[7000:]
test_label = labels[7000:]

len(train_data)
len(test_data)
len(train_label)
len(test_label)

# we need to create total bag of words as we are analyzing twitter sentiment stopwards need to be deleted but in a controlled fashion
# create bigram check that also if it's present there we can't delete stop wards ,

tknzr = TweetTokenizer()

vocabulary = [x.lower() for sent in train_data for x in tknzr.tokenize(sent)]

vocabulary

# now the bag of words are created we remove the stopwords now

stop_words = stopwords.words('english')

custom_stop_words = stop_words[:(stop_words.index('just') + 1)]

punclist = ["(", ")", ",", "[", "]", ".", ";", "-", "/",
            "s", "i", "?", "``", ":", "'", "!", "..."]

custom_stop_words.extend(punclist)

custom_stop_words

all_words = []

words_cleaned = [word for word in vocabulary
                 if 'http' not in word
                 and not word.startswith('@')
                 and not word.startswith('#')
                 and '@' not in word
                 and word != 'RT'
                 and word not in custom_stop_words
                 and len(word) >= 3]

words_cleaned = nltk.FreqDist(words_cleaned)

print(words_cleaned.most_common(50))

len(words_cleaned.keys())

word_features = list(words_cleaned.keys())[:3000]

word_features

train_data


def get_unigram_features(data, vocab):
    fet_vec_all = []
    for tup in data:
        single_feat_vec = []
        sent = tup.lower()  # lowercasing the dataset
        for v in vocab:
            if sent.__contains__(v):
                single_feat_vec.append(1)
            else:
                single_feat_vec.append(0)
        fet_vec_all.append(single_feat_vec)
    return fet_vec_all


def get_vader_intensity(data):
    senti_score = []
    for sent in data:
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(sent)
        senti_score.append(scores['compound'])
    return senti_score


def merge_features(featureList1, featureList2):
    # For merging two features
    if featureList1 == []:
        return featureList2
    merged = []
    for i in range(len(featureList1)):
        m = featureList1[i] + featureList2[i]
        merged.append(m)
    return merged


def get_lables(data):
    labels = []
    for tup in data:
        if tup == "neg":
            labels.append(-1)
        else:
            labels.append(1)
    return labels


def real_time_test(classifier, vocab):
    print("Enter a sentence: ")
    inp = input()
    print(inp)
    feat_vec_uni = get_unigram_features(inp, vocab)
    feat_vec_swn = get_senti_wordnet_features(test_data)
    feat_vec = merge_features(feat_vec_uni, feat_vec_swn)

    predict = classifier.predict(feat_vec)
    if predict[0] == 1:
        print("The sentiment expressed is: positive")
    else:
        print("The sentiment expressed is: negative")


def calculate_precision(prediction, actual):
    prediction = list(prediction)
    correct_labels = [predictions[i] for i in range(
        len(predictions)) if actual[i] == predictions[i]]
    precision = float(len(correct_labels)) / float(len(prediction))
    return precision


def real_time_test(classifier, vocab):
    print("Enter a sentence: ")
    inp = input()
    print(inp)
    feat_vec_uni = get_unigram_features(inp, vocab)
    feat_vec_swn = get_senti_wordnet_features(test_data)
    feat_vec = merge_features(feat_vec_uni, feat_vec_swn)

    predict = classifier.predict(feat_vec)
    if predict[0] == 1:
        print("The sentiment expressed is: positive")
    else:
        print("The sentiment expressed is: negative")


training_unigram_features = get_unigram_features(
    train_data, words_cleaned)  # vocabulary extracted in the beginning
training_vader_features = get_vader_intensity(train_data)

training_vader_features = [[x] for x in training_vader_features]


training_features = merge_features(
    training_unigram_features, training_vader_features)

training_labels = get_lables(train_label)

test_unigram_features = get_unigram_features(test_data, words_cleaned)
test_vader_features = get_vader_intensity(test_data)

test_vader_features = [[x] for x in test_vader_features]

test_features = merge_features(test_unigram_features, test_vader_features)

test_gold_labels = get_lables(test_label)


nb_classifier = GaussianNB().fit(
    training_features, training_labels)  # training process
predictions = nb_classifier.predict(test_features)

print("Precision of NB classifier is")
predictions = nb_classifier.predict(training_features)
precision = calculate_precision(predictions, training_labels)
print("Training data\t" + str(precision))
predictions = nb_classifier.predict(test_features)
precision = calculate_precision(predictions, test_gold_labels)
print("Test data\t" + str(precision))


svm_classifier = LinearSVC(penalty='l2', C=0.01).fit(
    training_features, training_labels)
predictions = svm_classifier.predict(training_features)

print("Precision of linear SVM classifier is:")
precision = calculate_precision(predictions, training_labels)
print("Training data\t" + str(precision))
predictions = svm_classifier.predict(test_features)
precision = calculate_precision(predictions, test_gold_labels)
print("Test data\t" + str(precision))

decision_classifier = DecisionTreeClassifier().fit(
    training_features, training_labels)
predictions = decision_classifier.predict(training_features)

print("Precision of linear SVM classifier is:")
precision = calculate_precision(predictions, training_labels)
print("Training data\t" + str(precision))
predictions = decision_classifier.predict(test_features)
precision = calculate_precision(predictions, test_gold_labels)
print("Test data\t" + str(precision))
