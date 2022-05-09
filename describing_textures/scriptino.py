import json
import random

# Count phrases frequency
# * return couples (phrase, count)
def phrase_freq(file):
    phrase_list = {}
    for row in file:
        for desc in row['descriptions']:
            phrases = desc.split(",")
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
            for word in desc.replace(",", "").split(' '):
                if word in word_list:
                    word_list[word] += 1
                else:
                    word_list[word] = 1
    return sorted(word_list.items(), key=lambda x:x[1], reverse=True)


# Count phrases frequency on training set
# * filter couples where count<2
# * return couples (phrase, count)
def phrase_freq_train (file, file_train):
    train_set = file_train['train']
    phrase_list = {}
    for row in file:
        if row['image_name'] in train_set:
            for desc in row['descriptions']:
                phrases = desc.split(",")
                for phrase in phrases:
                    if phrase in phrase_list:
                        phrase_list[phrase.strip()] += 1
                    else:
                        phrase_list[phrase.strip()] = 1
        else:
            continue
    return filter(lambda pair: pair[1] >= 2, sorted(phrase_list.items(), key=lambda x:x[1], reverse=True))


# Count words frequency on training set
# * filter couples where count<2
# * return couples (word, count)
def words_freq_train (file, file_train):
    train_set = file_train['train']
    word_list = {}
    for row in file:
        if row['image_name'] in train_set:
            for desc in row['descriptions']:
                for word in desc.replace(",", "").split(' '):
                    if word in word_list:
                        word_list[word] += 1
                    else:
                        word_list[word] = 1
        else:
            continue
    return filter(lambda pair: pair[1] >= 2, sorted(word_list.items(), key=lambda x:x[1], reverse=True))


# split image-descriptions file in:
# * training set // 60%
# * validation set // 15%
# * test set // 25%
def split(tr, te):
    file = open('./data_api/data/image_labels_1-130.json')
    data = json.load(file)
    # collect all image's 'image_name' and shuffle the array
    all = [img['image_name'] for img in data]
    #for img in data:all.append(img['image_name'])
    random.shuffle(all)
    # split all into train, test and validation sets
    
    train = all[:int(len(all)*tr)]
    if tr == 1 and te == 1:
        test = train.copy()
        val = train.copy()
    else:
        test = all[int(len(all)*int(tr)):int(len(all))*int(tr+te)]
        val = all[int(len(all))*int(tr+te):]
    # create dictionary of 'image_name' images
    # according to the sets split
    dict = {"test": test, "val": val, "train": train}
    # write the dictionary on a json file and return
    fd = open('./data_api/data/image_splits_d2t.json', 'w')
    fd.write(str(dict))
    fd.close()
    return json.dumps(dict)

def generate_descriptions(file):
    # image_name
    # details
    # edges
    # color_saturation
    # color_shades
    # background
    # single_instance
    # text
    # texture
    # perspective
    # return {"image_name": "/path", "category": "", "description": ["...details", "... edges", "..."]}
    list_obj = []
    for row in file:
        description = []
        for key in row:
            if key != "image_name":
                description.append(row[key])
        current_obj = {"image_name": row["image_name"], "category": row["image_name"].split("/")[0], "descriptions": description}
        list_obj.append(current_obj)
    return json.dumps(list_obj)

def generate_frequencies (file):
    image_list = {}
    for row in file:
        if row['image_name'] in image_list:
            image_list[row['image_name']] += 1
        else:
            image_list[row['image_name']] = 1
    return sorted(image_list.items(), key=lambda x:x[1], reverse=True)



if __name__ == '__main__':
    # json_file = open('./data_api/data/image_descriptions_d2t.json', 'r')
    # json_file_training = open('./data_api/data/image_splits_d2t.json', 'r')
    image_frequencies = generate_frequencies(json.load(open('./data_api/data/image_labels_1-130.json', 'r')))
    fd = open("./data_api/data/image_frequencies.txt", "w")
    for pair in image_frequencies:
        fd.write(str(pair[0])+" : "+str(pair[1])+"\n")
    fd.close()
    
    image_descriptions = generate_descriptions(json.load(open('./data_api/data/image_labels_1-130.json', 'r')))
    fd = open('./data_api/data/image_descriptions_d2t.json', 'w')
    fd.write(str(image_descriptions))
    fd.close()

    image_splits = split(1, 1)
    fd = open('./data_api/data/image_splits_d2t.json', 'w')
    fd.write(str(image_splits))
    fd.close()

    data_file = json.load(open('./data_api/data/image_descriptions_d2t.json', 'r'))
    data_file_training = json.load(open('./data_api/data/image_splits_d2t.json', 'r'))
    # write phrases frequency into a txt
    # format phrase : freq
    file_phrase_freq = phrase_freq(data_file)
    fd = open("./data_api/data/phrase_freq_d2t.txt", "w")
    for pair in file_phrase_freq:
        fd.write(str(pair[0])+" : "+str(pair[1])+"\n")
    fd.close()

    # write words frequency into a txt
    # format word : freq
    file_word_freq = word_freq(data_file)
    fd = open("./data_api/data/word_freq_d2t.txt", "w")
    for pair in file_word_freq:
        fd.write(str(pair[0])+" : "+str(pair[1])+"\n")
    fd.close()

    # write training phrases frequency into a txt
    # format phrase : freq
    file_phrase_freq_train = phrase_freq_train(data_file, data_file_training)
    fd = open("./data_api/data/phrase_freq_train_d2t.txt", "w")
    for pair in file_phrase_freq_train:
        fd.write(str(pair[0])+" : "+str(pair[1])+"\n")
    fd.close()
    
    # write training words frequency into a txt
    # format word : freq
    file_words_freq_train = words_freq_train(data_file, data_file_training)
    fd = open("./data_api/data/word_freq_train_d2t.txt", "w")
    for pair in file_words_freq_train:
        fd.write(str(pair[0])+" : "+str(pair[1])+"\n")
    fd.close()