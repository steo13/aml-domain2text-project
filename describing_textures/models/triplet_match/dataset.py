import random
import torch.utils.data as data

from data_api.dataset_api import TextureDescriptionData
from models.layers.img_encoder import build_transforms

class TripletTrainData(data.Dataset, TextureDescriptionData):
    # Constructor
    # * split — kind of set under analysis – train/validation/test
    # * lang_input — kind of language input — phrase/description
    # * neg_img – whether negative image
    # * neg_lang – whether negative phrase/description
    # * img_transform – transformation on the images
    def __init__(self, split='train', lang_input='phrase', neg_img=True, neg_lang=True):
        data.Dataset.__init__(self)
        TextureDescriptionData.__init__(self, phid_format='str')
        self.split = split
        self.lang_input = lang_input
        self.neg_img = neg_img
        self.neg_lang = neg_lang
        self.img_transform = build_transforms(is_train=False)

        self.pos_pairs = list()
        # iterate over the images inside a split (train/test/val)
        # 1. extract image json data starting from the image name
        # 2. create couples of (image_idx, lang_input_idxs)
        #   - lang_input = phrase       ––– create list of tuples (image_index, phrase_id)
        #   - lang_input = description  ––– create list of tuples (image_index, descr_id)
        for img_i, img_name in enumerate(self.img_splits[self.split]):
            img_data = self.img_data_dict[img_name]    
            if self.lang_input == 'phrase':
                self.pos_pairs += [(img_i, ph) for ph in img_data['phrase_ids']]
            elif self.lang_input == 'description':
                self.pos_pairs += [(img_i, desc_idx) for desc_idx in range(len(img_data['descriptions']))]
            else:
                raise NotImplementedError
        # pos_pairs = [(image_id, phrase_id), ...]  OR  [(image_id, description_id), ...]
        return

    '''
    @ return length of tuples [(image_id, phrase_id/description_id), ...] list
    '''
    def __len__(self):
        return len(self.pos_pairs)


    '''
    Given an index, retrive the relative tuple (image_id, phrase_id/description_id) from 'self.pos_pairs'.
    Hence, collect the relative image, a negative image and a negative phrase/description
    @ return pos_img as image at pos pair index, pos_lang as phrase/description at pos pair index, neg_img and neg_lang
    '''
    def __getitem__(self, pair_i):
        if self.lang_input == 'phrase':
            # retrive couple (image_index, phrase_id)
            img_i, pos_lang = self.pos_pairs[pair_i]
            # retrive current couple image json data {image_name, category, descriptions, phrase_ids, image}
            pos_img_data = self.get_split_data(self.split, img_i, load_img=True)
            # collect RGB Image and apply transformation
            pos_img = pos_img_data['image']
            pos_img = self.img_transform(pos_img)

            neg_lang = None
            if self.neg_lang:
                while True:
                    # draw randomly between the selected split (train/test/val) phrases
                    # if the drawn phrase id is NOT in the phrase ids used for describing
                    # the current image, break! Hence, it will be the nagative phrase
                    neg_lang = random.choice(self.phrases)
                    if neg_lang not in pos_img_data['phrase_ids']:
                        break

            neg_img = None
            if self.neg_img:
                while True:
                    # draw randomly between the selected split (train/test/val) images (names)
                    # use the name to access self.img_data_dict and get the json data about it
                    # if the current phrase id is NOT in the phrase ids used for describing
                    # the random image, break! Hence, it will be the nagative image
                    neg_img_name = random.choice(self.img_splits[self.split])
                    neg_img_data = self.img_data_dict[neg_img_name]
                    if pos_lang not in neg_img_data['phrase_ids']:
                        break
                neg_img = self.load_img(neg_img_name)
                neg_img = self.img_transform(neg_img)

        else: # self.lang_input == 'descriptions'
             # retrive couple (image_index, description_id)
            img_i, desc_i = self.pos_pairs[pair_i]
            # retrive current couple image json data {image_name, category, descriptions, phrase_ids, image}
            pos_img_data = self.get_split_data(self.split, img_i, load_img=True)
            # collect RGB Image and apply transformation
            pos_img = pos_img_data['image']
            pos_img = self.img_transform(pos_img)
            # set 'pos_lang' by accessing the json data descriptions and selecting the 'desc_i'-one
            pos_lang = pos_img_data['descriptions'][desc_i]

            neg_lang = None
            if self.neg_lang:
                while True:
                    img_name = random.choice(self.img_splits[self.split])
                    if img_name == pos_img_data['image_name']:
                        continue
                    # draw randomly between the descriptions of 'img_name' image
                    # if the drawn description id is NOT in the descriptions used for describing
                    # the current image, break! Hence, it will be the nagative description
                    neg_lang = random.choice(self.img_data_dict[img_name]['descriptions'])
                    if neg_lang not in pos_img_data['descriptions']:
                        break

            neg_img = None
            if self.neg_img:
                while True:
                    # draw randomly between the selected split (train/test/val) images (names)
                    # use the name to access self.img_data_dict and get the json data about it
                    # if the current description is NOT in the descriptions used for describing
                    # the random image, break! Hence, it will be the nagative image
                    neg_img_name = random.choice(self.img_splits[self.split])
                    neg_img_data = self.img_data_dict[neg_img_name]
                    if pos_lang not in neg_img_data['descriptions']:
                        break
                neg_img = self.load_img(neg_img_name)
                neg_img = self.img_transform(neg_img)

        return pos_img, pos_lang, neg_img, neg_lang
