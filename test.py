import os
import torch
import torchvision.models as models
import torch.nn as nn
import cv2
import numpy as np
import torch.utils.data as Data

vgg16 = models.vgg16_bn(pretrained=True)
pretrained_dict = vgg16.state_dict()
vgg16.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
)
model = vgg16
model_dict = model.state_dict()
for param in model.parameters():
    param.requires_grad = False

pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.eval()
print(model)

#  can add a layer named fc without parameters

def remove_DS_Store(path):
    for item in path:
        try:
            if item.split('/')[-1] == '.DS_Store':
                path.remove(item)
        except OSError as e:
            print(e)
    return path

def feature_extraction(path):   # path should be the image folder path without '/' at the end
    image_list = os.listdir(path)
    image_list.sort()
    image_list = remove_DS_Store(image_list)
    data_batch = torch.zeros([image_list.__len__(), 3, 244, 244])
    counter = 0
    for image_name in image_list:
        image = cv2.imread(path + '/' + image_name)
        image = cv2.split(image)
        image = image[0]
        temp = torch.zeros([244, 244])
        temp[58: 186, 78: 166] = torch.Tensor(image)
        image = temp
        data_batch[counter, 0] = image
        data_batch[counter, 1] = image
        data_batch[counter, 2] = image
        # label_batch[counter] = int(name)
        counter += 1
    data_batch = data_batch / 256
    output = model(data_batch)
    output = torch.max(output, 1)[0]
    return output.data.numpy().astype(np.float64), path.split('/')[-4]


if __name__ == '__main__':

    # prepare the data
    path = '/Users/caojiahe/PycharmProjects/Gait_Recognition/Data/Normalized_img/DatasetB/gait_period/'
    name_list = os.listdir(path)
    name_list.sort()
    name_list = remove_DS_Store(name_list)
    sum = 0
    for name in name_list:
        sum = sum + os.listdir(path + name + '/' + 'nm-01/090').__len__()    # sum is sum of all photos

    # data_batch = torch.zeros([sum, 3, 244, 244])
    # label_batch = torch.zeros([sum])




    index = np.zeros([name_list.__len__(), 100])
    a = 0
    for name in name_list:
        image_name_list = os.listdir(path + name + '/nm-01/090')
        image_name_list.sort()
        data_batch = torch.zeros([image_name_list.__len__(), 3, 244, 244])
        counter = 0
        for image_name in image_name_list:
            image = cv2.imread(path + name + '/nm-01/090/' + image_name)
            image = cv2.split(image)
            image = image[0]
            temp = torch.zeros([244, 244])
            temp[58: 186, 78: 166] = torch.Tensor(image)
            image = temp
            data_batch[counter, 0] = image
            data_batch[counter, 1] = image
            data_batch[counter, 2] = image
            # label_batch[counter] = int(name)
            counter += 1
        data_batch = data_batch / 256
        output = model(data_batch)
        output = torch.max(output, 1)[0]
        output = output.detach().numpy()
        index[a, :output.size] = output
        a += 1
        print(a)

    np.save('./index_period.npy', index)



    # label_batch = torch.LongTensor(label_batch.numpy())
    #
    # torch_dataset = Data.TensorDataset(data_batch, label_batch)
    # train_loader = Data.dataloader(dataset=torch_dataset, batch_size=1, shuffle=True)



    # print(list(model.features.parameters())[0].shape)


    # print(list(model.classifier.parameters()))
    # print(model)


    # print(result)

    pass