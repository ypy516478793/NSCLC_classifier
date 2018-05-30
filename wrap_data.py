import os
import numpy as np

np.random.seed(123)

path = "/home/cougarnet.uh.edu/pyuan2/Downloads/data/Lung_Data/images/ill/original/"
files = os.listdir(path)
files.sort(key= lambda x: int(x.split('-')[-1].split(".")[0]))
num = 1
likely_normal = []
for file in files:
    id = int(file.split('-')[-1].split(".")[0])
    while id != num:
        # print(num)
        likely_normal.append(num)
        num += 1
    num += 1

likely_normal += list(range(num,423))


normal_path = "/home/cougarnet.uh.edu/pyuan2/Downloads/data/Lung_Data/images/normal/"
files = os.listdir(normal_path)
files.sort(key= lambda x: int(x.split('-')[-1].split(".")[0]))
normal = []
for file in files:
    normal.append(int(file.split('-')[-1].split(".")[0]))

print(normal)
weird = [x for x in likely_normal if x not in normal]
print(weird)

def get_label_dict():
    label_dict = {}
    label = "/home/cougarnet.uh.edu/pyuan2/Downloads/data/Lung_Data/Lung1.clinical.csv"
    file = open(label, 'r')
    lines = file.readlines()
    file.close()
    for line in lines:
        line = line.split(",")
        id = line[0].replace("\"", "")
        type = line[6].replace("\"", "")
        if "-" in id:
            label_dict[id] = type
    return label_dict

label_dict = get_label_dict()

def get_data(ill_people, normal_people):
    file = open("/home/cougarnet.uh.edu/pyuan2/Downloads/data/Lung_Data/bad_data.txt", "r")
    bad_data = [line.strip() for line in file]
    file.close()
    all_data = []
    all_label = []
    weird_data = []
    for patient in ill_people:
        if patient not in bad_data:
            all_data.append(ill_people[patient])
            all_label.append(label_dict[patient])
            if label_dict[patient] == 'NA':
                weird_data.append(patient)
    num_sample = len(all_data)
    indices = np.random.permutation(num_sample)
    training_idx, test_idx = indices[:int(0.8*num_sample)], indices[int(0.8*num_sample):]
    training_data, test_data = [all_data[i] for i in training_idx], [all_data[i] for i in test_idx]
    training_label, test_label = [all_label[i] for i in training_idx], [all_label[i] for i in test_idx]
    # training_data, test_data = all_data[training_idx, :], all_data[test_idx, :]
    # training_label, test_label = all_label[training_idx, :], all_label[test_idx, :]

    return training_data, training_label, test_data, test_label






print("")
