from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

from transformers     import AutoTokenizer

import re
import os
import sys
import numpy as np
import pandas as pd
import cv2
import tqdm
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def prepare_data(data):
    imgs, captions, captions_lens, class_ids, keys, wrong_caps, wrong_caps_len, wrong_cls_id = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    if 'bert' in cfg.TEXT.TEXT_MODEL:
        captions = {k: v[sorted_cap_indices].squeeze() for k,v in captions.items()}
    else:
        captions = {k: v[sorted_cap_indices] for k,v in captions.items()}
        captions = captions[sorted_cap_indices].squeeze()

    class_ids = class_ids[sorted_cap_indices].numpy()
    keys = [keys[i] for i in sorted_cap_indices.numpy()]

    if cfg.CUDA:
        if 'bert' in cfg.TEXT.TEXT_MODEL:
            captions = {k: Variable(v).cuda() for k,v in captions.items()}
        else: 
            captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        if 'bert' in cfg.TEXT.TEXT_MODEL:
            captions = {k: Variable(v) for k,v in captions.items()}
        else: 
            captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    ## 
    w_sorted_cap_lens, w_sorted_cap_indices = \
        torch.sort(wrong_caps_len, 0, True)

    if 'bert' in cfg.TEXT.TEXT_MODEL:
        wrong_caps = {k: v[sorted_cap_indices] for k,v in wrong_caps.items()}
    else:
        wrong_caps = wrong_caps[w_sorted_cap_indices].squeeze()
    wrong_cls_id = wrong_cls_id[w_sorted_cap_indices].numpy()

    if cfg.CUDA:
        if 'bert' in cfg.TEXT.TEXT_MODEL:
            wrong_caps = {k: Variable(v).cuda() for k,v in wrong_caps.items()}
        else: 
            wrong_caps = Variable(wrong_caps).cuda()
        w_sorted_cap_lens = Variable(w_sorted_cap_lens).cuda()
    else:
        if 'bert' in cfg.TEXT.TEXT_MODEL:
            wrong_caps = {k: Variable(v) for k,v in wrong_caps.items()}
        else: 
            wrong_caps = Variable(wrong_caps)
        w_sorted_cap_lens = Variable(w_sorted_cap_lens)

    return [real_imgs, captions, sorted_cap_lens,
            class_ids, keys, wrong_caps, w_sorted_cap_lens, wrong_cls_id]


def get_imgs(img_path, imsize,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size

    if transform is not None:
        img = transform(img)

    ret = []
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Scale(imsize[i])(img)
            else:
                re_img = img
            ret.append(normalize(re_img))

    return ret


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CHEXPERT_DATA_DIR = '/home/mars/chexpert/CheXpert-v1.0' 
VIEW_COL = "Frontal/Lateral"
PATH_COL = "Path"
SPLIT_COL = "DataSplit"
REPORT_COL = "Report Impression"


class ChexpertDataset(data.Dataset):
    def __init__(
            self, 
            data_dir=CHEXPERT_DATA_DIR,
            split='train',
            base_size=64,
            transform=None, 
            target_transform=None
        ):
        self.transform = transform

        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])   
        self.target_transform = target_transform   
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE 

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        split_dir = os.path.join(data_dir, split)

        self.path2sent = dict()
        self.to_remove = []

        csv_path = os.path.join(CHEXPERT_DATA_DIR, 'master_updated.csv') 
        self.df = pd.read_csv(csv_path)
        self.df[PATH_COL] = self.df[PATH_COL].apply(
            lambda x: os.path.join(CHEXPERT_DATA_DIR, '/'.join(x.split('/')[1:]))
        )
        self.df = self.df[self.df[VIEW_COL] == "Frontal"]

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words, self.path2sent = self.load_text_data(data_dir, split)

        #self.class_id = self.load_class_id(split_dir, len(self.filenames))
        # TODO: figure out what to use for class id
        self.class_id = self.df[self.df.Split == split]['No Finding'].tolist()

        self.number_example = len(self.filenames)

        # using bert instead of LSTM
        if 'bert' in cfg.TEXT.TEXT_MODEL:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT.TEXT_MODEL)
            self.idxtoword = {v:k for k,v in self.tokenizer.get_vocab().items()}


    def load_captions(self, split):

        df_split = self.df[self.df.Split == split]

        all_captions = []
        for idx, row in tqdm.tqdm(df_split.iterrows(), total=df_split.shape[0]):

            captions = row[REPORT_COL].replace('\n', ' ')

            splitter = re.compile('[0-9]+\.')
            captions = splitter.split(captions)
            captions = [point.split('.') for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            for cap in captions:
                if len(cap) == 0:
                    continue
                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(cap.lower())
                if len(tokens) == 0:
                    #print('cap', cap)
                    continue

                tokens_new = []
                for t in tokens:
                    t = t.encode('ascii', 'ignore').decode('ascii')
                    if len(t) > 0:
                        tokens_new.append(t)
                all_captions.append(tokens_new)
                study_sent.append(tokens_new)
                cnt += 1
                if cnt == self.embeddings_num:
                    break
            
            if len(study_sent) > 0:
                self.path2sent[row[PATH_COL]] = study_sent
            else:
                self.to_remove.append(row[PATH_COL])
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            test_captions_new.append(rev)

        path2sent_new = defaultdict(list)
        for path, sent in self.path2sent.items():
            for t in sent:
                rev = []
                for w in t:
                    if w in wordtoix:
                        rev.append(wordtoix[w])
                path2sent_new[path].append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword), path2sent_new]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions('train')
            test_captions = self.load_captions('valid')

            train_captions, test_captions, ixtoword, wordtoix, n_words, path2sent = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix, path2sent, self.to_remove], 
                             f, protocol=2
                            )
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                print("filepath", filepath)
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                path2sent = x[4]
                self.to_remove = x[5]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = self.df[self.df.Split == 'train'][PATH_COL].tolist()
        else:  # split=='test'
            captions = test_captions
            filenames = self.df[self.df.Split == 'valid'][PATH_COL].tolist()
        
        filenames = [f for f in filenames if f not in self.to_remove]

        return filenames, captions, ixtoword, wordtoix, n_words, path2sent

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding='latin1')
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, path):
        
        series_sents = self.path2sent[path]


        if len(series_sents) == 0:
            print(path)
            raise Exception('no sentence for path')

        sent_ix = random.randint(0, len(series_sents))
        sent = series_sents[sent_ix] 

        # for LSTM
        if 'bert' not in cfg.TEXT.TEXT_MODEL:
            # a list of indices for a sentence
            sent_caption = np.asarray(sent).astype('int64')
            if (sent_caption == 0).sum() > 0:
                print('ERROR: do not need END (0) token', sent_caption)
            num_words = len(sent_caption)
            x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
            x_len = num_words
            if num_words <= cfg.TEXT.WORDS_NUM:
                x[:num_words, 0] = sent_caption
            
            # TODO: maybe not completely random resample word 
            else:
                ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
                np.random.shuffle(ix)
                ix = ix[:cfg.TEXT.WORDS_NUM]
                ix = np.sort(ix)
                x[:, 0] = sent_caption[ix]
                x_len = cfg.TEXT.WORDS_NUM
            return x, x_len
        else: 
            sent = ' '.join([self.ixtoword[x] for x in sent])

            tokens = self.tokenizer(
                sent, return_tensors='pt', truncation=True, 
                padding='max_length', max_length=cfg.TEXT.WORDS_NUM
            )
            x_len = len([t for t in tokens['input_ids'][0] if t != 0])

            return tokens, x_len

    def get_imgs(self, img_path, imsize,
                transform=None, normalize=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images 
        x = self._resize_img(x, imsize[-1])    # TODO: double check 
        img = Image.fromarray(x).convert('RGB')

        if transform is not None:
            img = transform(img)

        ret = []
        if cfg.GAN.B_DCGAN:
            ret = [normalize(img)]
        else:
            for i in range(cfg.TREE.BRANCH_NUM):
                if i < (cfg.TREE.BRANCH_NUM - 1):
                    re_img = transforms.Scale(imsize[i])(img)
                else:
                    re_img = img
                ret.append(normalize(re_img))

        return ret

    def __getitem__(self, index):

        key = self.filenames[index] # NOTE: not used in trainer 
        cls_id = self.class_id[index]

        img_name = key
        imgs = self.get_imgs(
            img_name, self.imsize, self.transform, normalize=self.norm
        )

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        # randomly select a mismatch sentence
        wrong_idx = random.randint(0, len(self.filenames))
        wrong_path = self.filenames[wrong_idx]
        wrong_cls_id = self.class_id[wrong_idx]
        wrong_caps, wrong_cap_len = self.get_caption(wrong_path)

        return imgs, caps, cap_len, cls_id, key, wrong_caps, wrong_cap_len, wrong_cls_id

    def __len__(self):
        return len(self.filenames)

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        #Resizing
        if max_ind == 0:
            #image is heigher
            wpercent = (scale / float(size[0]))
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            #image is wider
            hpercent = (scale / float(size[1]))
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(img, desireable_size[::-1], interpolation = cv2.INTER_AREA) #this flips the desireable_size vector

        #Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size/2))
            right = int(np.ceil(pad_size/2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size/2))
            bottom = int(np.ceil(pad_size/2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(resized_img,[(top, bottom), (left, right)], 'constant', constant_values=0)

        return resized_img