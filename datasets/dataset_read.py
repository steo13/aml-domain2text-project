import sys
sys.path.append('../loader')
from datasets.datasets_ import Dataset
import torchvision.transforms as transforms
import torch

'''
Read txt files line-by-line such as: PACS/kfold/art_painting/dog/pic_001.jpg 0
@ return file_names[] list containing paths
@ return labels[] list containing class 0 (dog) / 1 (elephant) / 2 (giraffe) / 3 (guitar) / ...
'''
def _dataset_info(args,txt_labels):

    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        # row = PACS/kfold/art_painting/dog/pic_001.jpg 0
        row = row.split(' ')
        # row[0] = PACS/kfold/art_painting/dog/pic_001.jpg
        file_names.append(args.path_to_dataset+row[0])
        # row[1] = 0
        labels.append(int(row[1]))

    return file_names, labels

'''
@ return image transformer
- resize 256x256
- center 224x224
- to tensor form
- normalize content (resnet optimization)
'''
def get_test_transformers():
    img_tr = [transforms.Resize(256),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr)


'''
@ return DataLoader starting from the Dataset created from file_names[], labels[] lists
'''
def dataset_read_eval(target,args):
    img_transformer = get_test_transformers()
    name_train,labels_train = _dataset_info(args,target)
    dataset = Dataset(name_train, labels_train, img_transformer=img_transformer)
    target_test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    return target_test_loader


    