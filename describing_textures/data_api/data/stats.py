import json
import random
from math import isclose
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

# Path for all the files
IMAGE_LABELS = './describing_textures/data_api/data/image_labels_1-100.json'
IMAGE_FREQUENCIES = './describing_textures/data_api/data/stats/image_frequencies.txt'
IMAGE_DESCRIPTIONS = './describing_textures/data_api/data/image_descriptions.json'
IMAGE_SPLITS = './describing_textures/data_api/data/image_splits.json'
PHRASE_FREQUENCY = './describing_textures/data_api/data/stats/phrase_freq_d2t.txt'
WORD_FREQUENCY = './describing_textures/data_api/data/stats/word_freq_d2t.txt'
PHRASE_FREQUENCY_TRAINING = './describing_textures/data_api/data/stats/phrase_freq_train_d2t.txt'
WORD_FREQUENCY_TRAINING = './describing_textures/data_api/data/stats/word_freq_train_d2t.txt'

# Percentage of the split for the dataset images
TRAIN = 1
TEST = 0
VAL = 0

# Count phrases frequency
# * return couples (phrase, count)
def phrase_freq(file):
    phrase_list = {}
    for row in file:
        for desc in row['descriptions']:
            phrases = desc.split(',')
            for phrase in phrases:
                if phrase.strip() in phrase_list:
                    phrase_list[phrase.strip()] += 1
                else:
                    phrase_list[phrase.strip()] = 1
    return sorted(phrase_list.items(), key=lambda x:x[1], reverse=True)


# Count words frequency
# * return couples (word, count)
def word_freq(file):
    word_list = {}
    for row in file:
        for desc in row['descriptions']:
            for word in desc.replace(',', '').split(' '):
                if word in word_list:
                    word_list[word] += 1
                else:
                    word_list[word] = 1
    return sorted(word_list.items(), key=lambda x:x[1], reverse=True)


# Count phrases frequency on training set
# * return couples (phrase, count)
def phrase_freq_train (file, file_train):
    train_set = file_train['train']
    phrase_list = {}
    for row in file:
        if row['image_name'] in train_set:
            for desc in row['descriptions']:
                phrases = desc.split(',')
                for phrase in phrases:
                    if phrase in phrase_list:
                        phrase_list[phrase.strip()] += 1
                    else:
                        phrase_list[phrase.strip()] = 1
        else:
            continue
    return filter(lambda pair: pair[1], sorted(phrase_list.items(), key=lambda x:x[1], reverse=True))


# Count words frequency on training set
# * return couples (word, count)
def words_freq_train (file, file_train):
    train_set = file_train['train']
    word_list = {}
    for row in file:
        if row['image_name'] in train_set:
            for desc in row['descriptions']:
                for word in desc.replace(',', '').split(' '):
                    if word in word_list:
                        word_list[word] += 1
                    else:
                        word_list[word] = 1
        else:
            continue
    return filter(lambda pair: pair[1], sorted(word_list.items(), key=lambda x:x[1], reverse=True))


# split image-descriptions file in:
# * training set 
# * validation set 
# * test set 
def generate_splits(tr, te, val):
    if not isclose(1, tr+te+val):
        print('Check the splits parameters, the sum is', tr+te+val)
        return
    else:
        file = open(IMAGE_LABELS)
        data = json.load(file)
        # collect all image's 'image_name' and shuffle the array
        all = [img['image_name'] for img in data]
        #for img in data:all.append(img['image_name'])
        random.shuffle(all)
        # split all into train, test and validation sets
        
        train = all[:int(len(all)*tr)]
        test = all[int(len(all)*tr):int(len(all)*(tr+te))]
        val = all[int(len(all)*(tr+te)):]
        # create dictionary of 'image_name' images
        # according to the sets split
        dict = {'test': test, 'val': val, 'train': train}

        return json.dumps(dict)

# Generate the file with all the descriptions of the images
# * return {'image_name': '/path', 'category': '', 'description': ['...details', '... edges', '...']}
def generate_descriptions(file, version='default'):
    list_obj = []
    for row in file:
        description = []
        for key in row:
            if key != 'image_name':
                description.append(row[key])
        current_obj = {'image_name': row['image_name'], 'category': row['image_name'].split('/')[0], 'descriptions': description}
        list_obj.append(current_obj)
    if version == 'default':
        return json.dumps(list_obj)
    else:
        return list_obj

# Count the images in the file
# * return the image list sorted by the frequency
def generate_frequencies (file):
    image_list = {}
    for row in file:
        if row['image_name'] in image_list:
            image_list[row['image_name']] += 1
        else:
            image_list[row['image_name']] = 1
    return sorted(image_list.items(), key=lambda x:x[1], reverse=True)

# Count words frequency
# * return couples (word, count)
def words_freq_for_cloud(_descriptions_):
    word_list = {
        "sketch": {},
        "cartoon": {},
        "photo": {},
        "art_painting": {}
    }
    for row in _descriptions_:
        category = row["category"]
        for desc in row['descriptions']:
            for word in desc.replace(',', '').split(' '):
                if word in word_list[category]:
                    word_list[category][word] += 1
                else:
                    word_list[category][word] = 1
    return word_list

# Generate the wordcloud for each domain
def gen_word_cloud(words_freqs):
    text = ''
    for word in words_freqs:
        text += word + ' '

    exep = {'il', 'lo', 'la', 'i', 'gli', 'le', 'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra', 'del', 'al',
            'dal', 'nel', 'col', 'sul', 'dello', 'allo', 'dallo', 'nello', 'sullo', 'della', 'alla', 'dalla', 'nella',
            'sulla', 'dell', 'all', 'dall', 'nell', 'sull', 'dei', 'ai', 'dai', 'nei', 'sui', 'coi', 'pei', 'degli',
            'agli', 'dagli', 'negli', 'sugli', 'delle', 'alle', 'dalle', 'nelle', 'sulle', 'ma', 'se', 'la', 'un',
            'una', 'ho', 'lo', 'vi', 'mi', 'degli', 'dei', 'co', 'del', 'chi', 'con', 'possono', 'essere', 'possono',
            'così', 'però', 'per', 'che', 'si', 'più', 'anche', 'cui', 'qui', 'perché', 'ad', 'con', 'cui', 'sono',
            'come', 'non', 'ha', 'quegli', 'quello', 'quella', 'quelli', 'io', 'qui', 'su', 'quel', 'qua', 'qui',
            'quando', 'ed', 'mentre', 'questo', 'può', 'abbiamo', 'siamo', 'questa', 'là', 'ci', 'https'}
    stopwords = set(STOPWORDS).union(exep)

    wordcloud_spa = WordCloud(stopwords=stopwords, background_color='white', max_words=2000).generate(text)
    plt.figure(figsize=[7, 7])
    plt.imshow(wordcloud_spa, interpolation='bilinear')
    plt.axis('off')
    plt.show()



if __name__ == '__main__':
    image_frequencies = generate_frequencies(json.load(open(IMAGE_LABELS, 'r')))
    fd = open(IMAGE_FREQUENCIES, 'w')
    for pair in image_frequencies:
        fd.write(str(pair[0])+' : '+str(pair[1])+'\n')
    fd.close()
    
    image_descriptions = generate_descriptions(json.load(open(IMAGE_LABELS, 'r')))
    fd = open(IMAGE_DESCRIPTIONS, 'w')
    fd.write(str(image_descriptions))
    fd.close()

    image_splits = generate_splits(TRAIN, TEST, VAL)
    fd = open(IMAGE_SPLITS, 'w')
    fd.write(str(image_splits))
    fd.close()

    file_descriptions = json.load(open(IMAGE_DESCRIPTIONS, 'r'))
    file_splits = json.load(open(IMAGE_SPLITS, 'r'))

    # write phrases frequency into a txt
    # format phrase : freq
    image_phrase_freq = phrase_freq(file_descriptions)
    fd = open(PHRASE_FREQUENCY, 'w')
    for pair in image_phrase_freq:
        fd.write(str(pair[0])+' : '+str(pair[1])+'\n')
    fd.close()

    # write words frequency into a txt
    # format word : freq
    image_word_freq = word_freq(file_descriptions)
    fd = open(WORD_FREQUENCY, 'w')
    for pair in image_word_freq:
        fd.write(str(pair[0])+' : '+str(pair[1])+'\n')
    fd.close()

    # write training phrases frequency into a txt
    # format phrase : freq
    image_phrase_freq_train = phrase_freq_train(file_descriptions, file_splits)
    fd = open(PHRASE_FREQUENCY_TRAINING, 'w')
    for pair in image_phrase_freq_train:
        fd.write(str(pair[0])+' : '+str(pair[1])+'\n')
    fd.close()
    
    # write training words frequency into a txt
    # format word : freq
    image_word_freq_train = words_freq_train(file_descriptions, file_splits)
    fd = open(WORD_FREQUENCY_TRAINING, 'w')
    for pair in image_word_freq_train:
        fd.write(str(pair[0])+' : '+str(pair[1])+'\n')
    fd.close()

    fd = open(IMAGE_LABELS)
    data = json.load(fd)
    descriptions = generate_descriptions(data, 'cloud')
    frequencies = words_freq_for_cloud(descriptions)
    for domain in frequencies:
        gen_word_cloud(frequencies[domain])