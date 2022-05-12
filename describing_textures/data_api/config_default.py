from yacs.config import CfgNode as CN

C = CN()

C.FILE_DESCRIPTIONS = 'image_descriptions_d2t.json'

C.PHRASE_FREQ_FILE = 'phrase_freq.txt'
C.PHRASE_FREQ_TRAIN_FILE = 'phrase_freq_train.txt'
C.WORD_FREQ_FILE = 'word_freq.txt'
C.K = [5, 10, 20, 50, 100]