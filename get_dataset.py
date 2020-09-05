import os
import cv2
import json
import numpy as np
import random


def celaba():
    id_txt = './identity_CelebA.txt'
    dataset_dir = './celeba'
    dest_dir = './data'
    # types = [os.path.join(dataset_dir, i) for i in ['train', 'test', 'eval']]
    dicts = {}
    label_ids = [4119, 6110, 7910, 7300, 10174,
                 10045, 10021, 6311, 9007, 5866,
                 5998, 10014, 9927, 10093, 10085,
                 8943, 10050, 2014, 6151, 5805]

    # label_ids = [str(i) for i in label_ids]

    with open(id_txt, 'r') as f:
        for line in f.readlines():
            name, id = line.split(' ')
            if eval(id) in label_ids:
                names = dicts.get(id, [])
                # print(name, id, names)
                names.append(name)
                dicts[id] = names
    print(dicts)

    with open('data.json', 'w') as f:
        json.dump(dicts, f)

    for (label, data) in dicts.items():
        print(label)
        class_path = os.path.join(dest_dir, str(eval(label)))
        print(class_path)
        if not os.path.exists(class_path):
            os.makedirs(class_path)
        for i in range(len(data)):
            os.popen('cp {0} {1}'.format(os.path.join(dataset_dir, data[i]), os.path.join(class_path, data[i])))


def cal_transform(data_dir):
    # BGR, facescrub
    # means: [0.39632604, 0.45772487, 0.59645426]
    # stdevs: [0.2273719, 0.23214313, 0.27194658]
    # RGB, celeba
    # mean=[0.4944, 0.4020, 0.3556], std=[0.3068, 0.2830, 0.2780]

    imgs = np.zeros([64, 64, 3, 1])
    means, stdevs = [], []

    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        # dest_dir = os.path.join(new_dir, label)
        # if not os._exists(dest_dir):
        #     os.makedirs(dest_dir)
        images = os.listdir(label_dir)
        for i in range(3):
            img_name = images[i]
        # for img_name in os.listdir(label_dir):
            img = cv2.imread(os.path.join(label_dir, img_name))
            # img = img[20:198, :]
            # img = cv2.resize(img, (64, 64))
            # cv2.imwrite(os.path.join(dest_dir, img_name), img)
            img = img[:, :, :, np.newaxis]
            imgs = np.concatenate((imgs, img), axis=3)

    imgs = imgs.astype(np.float32)/255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    print("means: ", means)
    print("stdevs: ", stdevs)

    means.reverse()
    stdevs.reverse()

    print("new means: ", means)
    print("stdevs: ", stdevs)


def facescrub(data_dir, dest_dir):
    for person in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person)
        person_dir = os.path.join(person_dir, 'face')
        for img_name in os.listdir(person_dir):
            try:
                img = cv2.imread(os.path.join(person_dir, img_name))
                print(person, img_name)
                img = cv2.resize(img, (64, 64))
                img_dest_path = os.path.join(dest_dir, person)
                if not os.path.exists(img_dest_path):
                    os.makedirs(img_dest_path)
                cv2.imwrite(os.path.join(img_dest_path, img_name), img)
            except:
                pass


cal_transform('./data_facescrub')