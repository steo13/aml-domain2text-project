import json
import os
import random
import re
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import sys
sys.path.append('/content/aml-domain2text-project/describing_textures')
from data_api.config_default import C as cfg

data_path = './aml-domain2text-project/describing_textures/data_api/data'
img_path = './aml-domain2text-project/describing_textures/data_api/data/images'

########################################################
##                   DESCRIPTION DATASET              ##
# ######################################################

class TextureDescriptionData:
    def __init__(self, phrase_split='train', phrase_freq_thresh=10, phid_format='set'):
        self_path = os.path.realpath(__file__)
        self_dir = os.path.dirname(self_path)
        self.data_path = os.path.join(self_dir, 'data')

        #with open(os.path.join(self.data_path, 'image_splits.json'), 'r') as f:
        # load images path beloging to the set under analysis
        with open(os.path.join(self.data_path, cfg.FILE_DESCRIPTIONS), 'r') as f:
            self.img_splits = json.load(f)

        # set if working on whole dataset or training set
        self.phrases = list()
        self.phrase_freq = list()
        if phrase_split == 'all':
            phrase_freq_file = cfg.PHRASE_FREQ_FILE
        elif phrase_split == 'train':
            phrase_freq_file = cfg.PHRASE_FREQ_FILE_TRAIN
        else:
            raise NotImplementedError
        
        # open <phrase,frequency> txt and read it line-by-line
        # 1. read current line
        # 2. extract phrase and count splitting the line
        # 3. filter reads where count < threshold (10 by default)
        # 4. collect separately phrase and count
        with open(os.path.join(self.data_path, phrase_freq_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                phrase, freq = line.split(' : ')
                if int(freq) < phrase_freq_thresh:
                    break
                self.phrases.append(phrase)
                self.phrase_freq.append(int(freq))

        # enumerate each phrase with a unique incremental id
        # dict: {'phrase': id, ...}
        self.phrase_phid_dict = {p: i for i, p in enumerate(self.phrases)}
        # get phrase id format (set by default)
        self.phid_format = phid_format

        # create a dictionary called 'img_data_dict' for the images like
        # {"image_name" : {phrase_ids: [id_description1, ...], ...}, ...}
        self.img_data_dict = dict()
        with open(os.path.join(self.data_path, cfg.FILE_DESCRIPTIONS), 'r') as f:
            # load data as ARRAY of objects like
            #  {
            #   "image_name": "sketch/giraffe/n02439033_10919-5.png",
            #   "category": "sketch",
            #   "descriptions": ["low-level details", "evident outlines", ...]
            # }
            data = json.load(f)
        for img_d in data:
            # get descriptions phrases ids for the current image
            # img_d['phrase_ids'] = [[desc1_id1, desc1_sid2, desc1_idn], [descX_id1, descX_sid2, descX_idn], ...]
            img_d['phrase_ids'] = self.descpritions_to_phids(img_d['descriptions'])
            # 
            # since img_d['image_name']=path, in the end, you'll get something like
            #
            # img_data_dict = {
            #   "sketch/giraffe/n02439033_10919-5.png": {
            #       "image_name": "sketch/giraffe/n02439033_10919-5.png",
            #       "category": "sketch",
            #       "descriptions": ["low-level details", "evident outlines, ...]
            #       "phrase_ids": [[desc1_id1, desc1_sid2, desc1_idn], [descX_id1, descX_sid2, descX_idn], ...]
            #   },
            #   ...
            # }
            self.img_data_dict[img_d['image_name']] = img_d

        self.img_phrase_match_matrices = dict()
        print('TextureDescriptionData ready. \n{ph_num} phrases with frequency above {freq}.\n'
              'Image count: train {train}, val {val}, test {test}'
              .format(ph_num=len(self.phrases), freq=phrase_freq_thresh, train=len(self.img_splits['train']),
                      val=len(self.img_splits['val']), test=len(self.img_splits['test'])))


    '''
    Return data related to an image, given a set (train/val/test) and
    directly accessing to an image (random, by default) using an index
    @ return json data with {image_name, category, descriptions, phrase_ids, image}
    '''
    def get_split_data(self, split=None, img_idx=None, load_img=False):
        if split is None:
            # whole set of img_names composed by train+val+test ones
            img_names = self.img_splits['train'] + self.img_splits['val'] + self.img_splits['test']
        else:
            # sheer paranoia: is split one between train, test or val? 
            # if so ok, otherwise assert it!
            assert split in self.img_splits
            # collect inside 'img_names' (*) images paths of the given split
            img_names = self.img_splits[split]

        if img_idx is None:
            # if 'img_idx' not specified, draw randomly from 'img_names'(*)
            img_name = random.choice(img_names)
        else:
            # sheer paranoia: is the given index not out-of-range?
            # if so ok, otherwise assert it!
            assert img_idx < len(img_names)
            # access directly on 'img_names'(*) and get relative image's path 'img_name' (◊)
            img_name = img_names[img_idx]

        # access 'img_data_dict' via image_name(◊) and and get the relative json data
        # img_data_dict is something like <image_name, {image_name, category, descriptions, phrase_ids}>     
        img_data = self.img_data_dict[img_name]
        if not load_img:
            return img_data
        # add to standard 'image_data' json the key-value association <"image": RGB Image>
        # and return that "improved" json
        img_data['image'] = self.load_img(img_name)
        return img_data


    '''
    Retrive image via relative datapath and return RGB Image
    @ return Image object of the current image
    '''
    def load_img(self, img_name):
        img_fpath = os.path.join(self.data_path, 'images', img_name)
        img = Image.open(img_fpath).convert('RGB')
        return img


    '''
    Given a phraseid, return relative textual phrase if known, otherwhise return string <UNK>
    @ return textual phrase given current phraseid
    '''
    def phid_to_phrase(self, phid):
        # phid = index used for direct access
        # on phrases [phrase1, phrase2, phrase3, ..., phraseN]
        if phid > len(self.phrases) or phid < 0:
            return '<UNK>'
        return self.phrases[phid]


    '''
    Return the value (id) starting from the the given key (phrase) if appears
    in the dictionary phrase_phid_dict, which is like {'phrase1': id1, 'phrase2': id2, ...}
    @ return id of the passed phrase
    '''
    def phrase_to_phid(self, phrase):
        return self.phrase_phid_dict.get(phrase, -1)


    '''
    Return a list of phrases composing a description
    @ return array of textual phrases
    '''
    @staticmethod
    def description_to_phrases(desc):
        segments = re.split('[,;]', desc)
        phrases = list()
        for seg in segments:
            phrase = seg.strip()
            if len(phrase) > 0:
                phrases.append(phrase)
        return phrases


    '''
    Return phrase ids starting from the image descriptions
    @ return depends on phid_format argument
        - str           set of whole textual phrases
        - nested_list   list of lists, each one composed by the numerical phraseids acquired from each description ([descX_id1, descX_id2, descX_idn])
        - phid_freq     dictionary of <"phraseid": frequency>
        - set           set of lists each one composed by the numerical phraseids acquired from each description ([descX_id1, descX_id2, descX_idn])
    '''
    def descpritions_to_phids(self, descriptions, phid_format=None):
        if phid_format is None:
            phid_format = self.phid_format

        if phid_format is None:
            return None
        
        # 'str' format —— description >> phrases
        phrases = set()
        if phid_format == 'str':
            for desc in descriptions:
                phrases.update(self.description_to_phrases(desc))
            return phrases

        # make a list of ids related to the single phrases
        # strarting from descriptions such as phids = [[id1, id2, ...], [...]]
        phids = list()
        for desc in descriptions:
            # get list of phrases from a single description
            phrases = self.description_to_phrases(desc)
            # given the current description 'desc', for each of its phrase
            # get the associated numeric id and append to 'phids_desc'
            phids_desc = [self.phrase_to_phid(ph) for ph in phrases]
            # append the current description phraseids list
            # [[desc1_id1, desc1_id2, desc1_idn], [desc2_id1, desc2_id2, desc2_idn], ...]
            phids.append(phids_desc)

        # return as list of lists
        # [[desc1_id1, desc1_id2, desc1_idn], [desc2_id1, desc2_id2, desc2_idn], ...]
        if phid_format == 'nested_list':
            return phids

        # make a dictionary <phraseid, frequency> starting from 'phids' list
        # which is something like [[desc1_id1, desc1_id2, desc1_idn], [desc2_id1, desc2_id2, desc2_idn], ...]
        elif phid_format == 'phid_freq':
            phid_freq = dict()
            for phids_desc in phids:
                # current 'phids_desc' = [descX_id1, descX_id2, descX_idn]
                for phid in phids_desc:
                    # phid = descX_idm
                    phid_freq[phid] = phid_freq.get(phid, 0) + 1
            return phid_freq

        # make a set of lists, each one containing phrases ids, starting from 'phids' list
        # which is something like [[desc1_id1, desc1_id2, desc1_idn], [desc2_id1, desc2_id2, desc2_idn], ...]
        elif phid_format == 'set':
            phid_set = set()
            for phids_desc in phids:
                # current 'phids_desc' = [descX_id1, descX_id2, descX_idn]
                phid_set.update(phids_desc)
            return phid_set

        else:
            raise NotImplementedError


    '''
    Given a certain description 'desc', return its phrase ids
    @ return set of phrase ids
    '''
    def description_to_phids_smart(self, desc):
        phids = set()
        # get textual phrases from the given description 
        phrases = self.description_to_phrases(desc)
        # iterate over the phrases
        # 1. if current phrase is in the image descriptions phrases
        #   - get the current phrase numerical id
        #   - append it to the phids set
        # 2. if current phrase is NOT in the image descriptions phrases
        #   - tokenize the current phrase
        #   - iterate over the words and append to set phids if current word is in the image descriptions phrases
        for ph in phrases:
            if ph in self.phrases:
                phids.add(self.phrase_to_phid(ph))
            else:
                for wd in WordEncoder.tokenize(ph):
                    if wd in self.phrases:
                        phids.add(self.phrase_to_phid(wd))
        return phids


    '''
    Given the split (train/test/val) fill/return the relative img_phrase matrix, where #rows = #images for that split
    and #cols = #phrases for the whole split. Given a row (image), this matrix is filled by 1 where the phrase (id)
    is used for describing that image. Fon instance:

                                            PHID    PHID    PHID   
                                        IMG  0       1       0 
                                        IMG  1       1       1
                                        IMG  0       0       1

    @ return numpy matrix, which tracks which phrases are used for describing the split images.
    '''
    def get_img_phrase_match_matrices(self, split):
        # img_phrase_match_matrices = <split, array>
        if split in self.img_phrase_match_matrices:
            return self.img_phrase_match_matrices[split]
        # get how many images inside the split
        # get how many phrases inside the split
        img_num = len(self.img_splits[split])
        phrase_num = len(self.phrases)
        # create a matrix of zeros to be filled
        # * n_rows = number of images
        # * n_cols = number of phrases
        match = np.zeros((img_num, phrase_num), dtype=int)
        # iterate over the image names
        # 1. 'img_data' as the json for the current image <image_name, {image_name, category, descriptions, phrase_ids}>
        # 2. 'phid_set' as the phrase_ids list [descX_id1, descX_id2, descX_idn] for the current image
        # 3. iterate over the phrases ids for the current description list [descX_id1, descX_id2, descX_idn]
        #   -  fill match cell image_i as row and phid as column with 1
        for img_i, img_name in enumerate(self.img_splits[split]):
            img_data = self.img_data_dict[img_name]
            if self.phid_format == 'set':
                phid_set = img_data['phrase_ids']
            else:
                phid_set = self.descpritions_to_phids(img_data['descriptions'], phid_format='set')
            for phid in phid_set:
                if phid >= 0:
                    match[img_i, phid] = 1
        # dictionary {"split": match}, where match is the matrix
        # filled with 1 where the phrase was used for describing that image
        self.img_phrase_match_matrices[split] = match
        return match


    '''
    Given a certain split (train/test/val) of images, return an
    array with #phrases for each image [#phrases_img1, #phrases_img2, ...]
    @ return an array of integers of the given split length
        - each cell refers to the relative image
        - each cell contains the #phrases for that image 
    '''
    def get_gt_phrase_count(self, split):
        # create an array of N zeros, where N is the given split (test/val/train) length
        gt_phrase_count = np.zeros(len(self.img_splits[split]))
        # iterate over that images name
        # 1. 'img_data' as the json for the current image <image_name, {image_name, category, descriptions, phrase_ids}>
        # 2. iterate over current image descriptions and collect inside 'phrases' set all the textual phrases that compose the image's descriptions
        # 3. access 'gt_phrase_count' using the iteration index and overwrite 0 with the number of phrases that describes the current image
        for img_i, img_name in enumerate(self.img_splits[split]):
            phrases = set()
            img_data = self.img_data_dict[img_name]
            for desc in img_data['descriptions']:
                phrases.update(self.description_to_phrases(desc))
            gt_phrase_count[img_i] = len(phrases)
        return gt_phrase_count



########################################################
##                   IMAGE DATASET                    ##
# ######################################################


class ImgOnlyDataset(Dataset):
    def __init__(self, split, transform=None, texture_dataset=None):
        Dataset.__init__(self)
        if texture_dataset is None:
            texture_dataset = TextureDescriptionData(phid_format=None)
        self.dataset = texture_dataset
        self.split = split
        self.transform = transform

    '''
    Get image item by index
    @ return (image_name, img) as image's path and RGB Image
    '''
    def __getitem__(self, idx):
        img_data = self.dataset.get_split_data(self.split, img_idx=idx, load_img=True)
        img_name = img_data['image_name']
        img = img_data['image']
        if self.transform is not None:
            img = self.transform(img)
        return img_name, img

    '''
    Given the self split (test/val/train) return the dataset len
    @ return dataset split (test/val/train) length 
    '''
    def __len__(self):
        return len(self.dataset.img_splits[self.split])



########################################################
##                   PHRASE DATASET                   ##
# ######################################################


class PhraseOnlyDataset(Dataset):
    def __init__(self, texture_dataset=None):
        Dataset.__init__(self)
        if texture_dataset is None:
            texture_dataset = TextureDescriptionData(phid_format=None)
        self.dataset = texture_dataset

    '''
    Get phrase item by index accessing directly inside the 'texture_dataset' phrases list
    @ return textual phrase
    '''
    def __getitem__(self, idx):
        return self.dataset.phrases[idx]

    '''
    Get the dataset len as #phrases inside the dataset
    @ return dataset length 
    '''
    def __len__(self):
        return len(self.dataset.phrases)



###############################################################
##                   WORD ENCODER                            ##
# Encorder for the words used inside the images descriptions  #
# #############################################################

class WordEncoder:
    def __init__(self, word_freq_file=cfg.WORD_FREQ_FILE, word_freq_thresh=5, special_chars=",;/&()-'"):
        self_path = os.path.realpath(__file__)
        self_dir = os.path.dirname(self_path)
        data_path = os.path.join(self_dir, 'data')

        self.word_list = ['<pad>']
        self.word_freq = [0]
        self.word_map = None

        # iterate over the txt file composed by couples <word : frequency>
        # 1. read line-by-line
        # 2. get separately word and count from the current line
        # 3. append word and count to the relative list
        with open(os.path.join(data_path, word_freq_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                word, freq = line.split(' : ')
                if int(freq) < word_freq_thresh:
                    break
                self.word_list.append(word)
                self.word_freq.append(int(freq))

        # append special chars to 'word_list' (",;/&()-'" by default)
        # stretch 'word_freq' adding 1 for each special char
        if special_chars is not None:
            self.word_list += [ch for ch in special_chars]
            self.word_freq += [1] * len(special_chars)
        self.word_list += ['<unk>', '<start>', '<end>']
        self.word_freq += [0, 0, 0]

        # return the word map {"word1": index1, "word2": index2, ...,}
        self.word_map = {w: idx for idx, w in enumerate(self.word_list)}

    '''
    Split an input phrase into single words (tokens)
    @ return list of words given the lang_input (phrase by default)
    '''
    @staticmethod
    def tokenize(lang_input):
        words = re.split('(\W)', lang_input)
        words = [w.strip() for w in words if len(w.strip()) > 0]
        return words


    '''
    Re-built a phrase starting from the relative list of words
    @ return caption of words
    '''
    def detokenize(self, tokens):
        caption = ' '.join(tokens)
        for ch in ',;':
            caption = caption.replace(' ' + ch + ' ', ch + ' ')
        for ch in "/()-'":
            caption = caption.replace(' ' + ch + ' ', ch)
        return caption


    '''
    Encode a 'lang_input' (phrase by default)
    @ return tuple (wordlist, length)
        - max_len<=0               encoded list and relative length
        - len(encoded)>max_len     truncated encoded list until max_len and relative length (max_len)
        - 0<max_len<max_len        encoded list (+padding until max_len) and relative length (without padding)
    '''
    def encode(self, lang_input, max_len=-1):
        # tokenize the input and add the <start> and the <end> tokens
        tokens = ['<start>'] + self.tokenize(lang_input) + ['<end>']
        # get encoded list as the word index if known inside the map, <unk> otherwhise
        # - for instance: [id1, id2, <unk>, id3, ..., <unk>, idn]
        encoded = [self.word_map.get(word, self.word_map['<unk>']) for word in tokens]
        if max_len <= 0:
            # defulat max_len returns (wordlist, length)
            return encoded, len(encoded)
        if len(encoded) >= max_len:
            # returns the wordlist untill max_len and max_len value
            return encoded[:max_len], max_len
        # get length of the worlist
        l = len(encoded)
        # pad the list (fill using <pad> token) until max_len is reached
        encoded += [self.word_map['<pad>']] * (max_len - len(encoded))
        # return worlist and length (without considering the padding!)
        return encoded, l


    '''
    Given the lang_inputs (phrases, by default), collect wordlists and relative lengths.
    Then, build a matrix in which #rows = #lists and #cols = length of the larger list (max_l),
    in which each cell contains the word for that wordlist. Zero remains as padding until max_l is reached.
    For instance:
                    0       1       2      3      4      MAX
        WLMAX     hey      men     how   are      you    buddy
        WL        hello    to      you     0      0       0
        WL        ehi      0       0       0      0       0
        WL        hello    buddy   0       0      0       0
    @ return wordlists matrix and list of lengths.
    '''
    def encode_pad(self, lang_inputs):
        encoded = list()
        lens = list()
        # iterate over the language input
        # 1. get worlist (encoded list) and relative length
        # 2. collect current wordlist and length
        for lang in lang_inputs:
            e, l = self.encode(lang, max_len=-1)
            encoded.append(e)
            lens.append(l)
        # get max length among the encoded lists lengths
        max_l = max(lens)
        # create a matrix of zeros
        # * n_rows = #encoded wordlists
        # * n_cols = max length (of the larger list)
        padded = np.zeros((len(encoded), max_l), dtype=np.long)
        # for each row (wordlist) fill the matrix with the words 
        # until the wordlist length is reached (0 remains until max_l)
        for i in range(len(encoded)):
            padded[i, :lens[i]] = encoded[i]
        return padded, lens
