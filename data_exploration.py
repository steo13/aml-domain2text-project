import json

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np

DESCRIPTIONS_PATH = './' # "./describing_textures/data_api/data/"
DESCRIPTIONS_JSON = "image_labels_1-100.json"


# Count words frequency
# * return couples (word, count)
def words_freq(_descriptions_):
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


# Generate the file with all the descriptions of the images
# * return {'image_name': '/path', 'category': '', 'description': ['...details', '... edges', '...']}
def generate_descriptions(file):
    list_obj = []
    for row in file:
        description = []
        for key in row:
            if key != 'image_name':
                description.append(row[key])
        current_obj = {'image_name': row['image_name'], 'category': row['image_name'].split('/')[0], 'descriptions': description}
        list_obj.append(current_obj)
    return list_obj


def genWordCloud(words_freqs):
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

    mask = np.array(Image.open('twitter.png'))
    wordcloud_spa = WordCloud(stopwords=stopwords, background_color='white', mask=mask, max_words=2000).generate(text)
    image_colors = ImageColorGenerator(mask)
    plt.figure(figsize=[7, 7])
    plt.imshow(wordcloud_spa.recolor(color_func=image_colors), interpolation='bilinear')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # Open JSON file
    f = open(DESCRIPTIONS_PATH+DESCRIPTIONS_JSON)
    # Return JSON object as a dictionary
    data = json.load(f)
    # Reduce descriptions
    descriptions = generate_descriptions(data)
    # Get word frequencies ["domain": {"word1": freq1, "word2": freq2, ...}]
    frequencies = words_freq(descriptions)
    # WordCloud for each domain
    for domain in frequencies:
        genWordCloud(frequencies[domain])

    # barplot source (X) vs distances (Y)
    # genWordCloud()
    # barplot for each domain
    # PCA(m=2) for each domain
    # embedding for each domain