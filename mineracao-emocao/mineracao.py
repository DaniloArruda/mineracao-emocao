import nltk
import base


stopwords = ['a', 'agora', 'algum', 'alguma', 'aquele', 'aqueles', 'de', 'deu', 'do', 'e', 'estou', 'esta', 'esta',
             'ir', 'meu', 'muito', 'mesmo', 'no', 'nossa', 'o', 'outro', 'para', 'que', 'sem', 'talvez', 'tem', 'tendo',
             'tenha', 'teve', 'tive', 'todo', 'um', 'uma', 'umas', 'uns', 'vou']

stop_words_nltk = nltk.corpus.stopwords.words('portuguese')


# apply stemming algorithm and remove the stop words.
def apply_stemming(base):
    phrases = []
    stemmer = nltk.stem.RSLPStemmer()

    for (phrase, emotion) in base:
        new_phrase = [str(stemmer.stem(word)) for word in phrase.split() if word not in stop_words_nltk]
        phrases.append((new_phrase, emotion))

    return phrases


# return all words in phrases
def get_words(phrases):
    words = []
    for (phrase, emotion) in phrases:
        words.extend(phrase)

    return words


# return the frequency of words in word vector
def get_frequency_of_words(words):
    return nltk.FreqDist(words)


# return distinct words
def get_distinct_words(frequency):
    return frequency.keys()


def get_feature_set(words):
    words = set(words)
    features = {}
    for word in distinct_words_training:
        features[word] = (word in words)
    return features


def get_errors(classifier, base_test):
    errors = []
    for (feature_set, emotion) in base_test:
        result = classifier.classify(feature_set)
        if result != emotion:
            errors.append((emotion, result, feature_set))
    return errors


def show_errors(classifier, base_test):
    errors = get_errors(classifier, base_test)
    for (emotion, result, feature_set) in errors:
        print(emotion, result, feature_set)


def show_confusion_matrix(classifier):
    from nltk.metrics import ConfusionMatrix
    expected = []
    resulted = []
    for (sentence, emotion) in final_base_test:
        result = classifier.classify(sentence)
        expected.append(emotion)
        resulted.append(result)

    matrix = ConfusionMatrix(expected, resulted)
    print(matrix)


def process_sentence(sentence):
    processed_sentence = []
    stemmer = nltk.stem.RSLPStemmer()
    for word in sentence.split():
        if word not in stop_words_nltk:
            processed_sentence.append(str(stemmer.stem(word)))
    return processed_sentence


def classify(classifier, sentence):
    processed_sentence = process_sentence(sentence)
    feature_set = get_feature_set(processed_sentence)

    return classifier.classify(feature_set)


def classify_with_probabilities(classifier, sentence):
    processed_sentence = process_sentence(sentence)
    feature_set = get_feature_set(processed_sentence)

    return classifier.prob_classify(feature_set)


# database pre-processing

stemming_sentence_training = apply_stemming(base.training)
stemming_sentence_test = apply_stemming(base.test)

words_training = get_words(stemming_sentence_training)
words_test = get_words(stemming_sentence_test)

words_frequency_training = get_frequency_of_words(words_training)
words_frequency_test = get_frequency_of_words(words_test)

distinct_words_training = get_distinct_words(words_frequency_training)
distinct_words_test = get_distinct_words(words_frequency_test)

# make feature set
final_base_training = nltk.classify.apply_features(get_feature_set, stemming_sentence_training)
final_base_test = nltk.classify.apply_features(get_feature_set, stemming_sentence_test)

# end database pre-processing


# classify

# make probability table: training
classifier = nltk.NaiveBayesClassifier.train(final_base_training)

print(nltk.classify.accuracy(classifier, final_base_test))

sentence = input('Digite uma frase: ')
classe = classify(classifier, sentence)
probabilities = classify_with_probabilities(classifier, sentence)

for sample in probabilities.samples():
    print('%s: %f' % (sample, probabilities.prob(sample)))



