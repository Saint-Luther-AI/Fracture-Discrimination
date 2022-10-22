import random
import numpy as np
import pandas as pd
import os
import h5py
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def lbp_3D(image, s): # 3D texture feature extraction

    z, w, h = image.shape
    texture_matrix = np.zeros([z, w, h])
    weight = [2 ** 0, 2 ** 1, 2 ** 2, 2 ** 3, 2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7,
              2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12, 2 ** 13, 2 ** 14,
              2 ** 15, 2 ** 16, 2 ** 17, 2 ** 18, 2 ** 19, 2 ** 20, 2 ** 21,
              2 ** 22, 2 ** 23, 2 ** 24, 2 ** 25]

    for k in range(s, z - s):
        for i in range(s, w - s):
            for j in range(s, h - s):
                binary_code = np.zeros(26)
                if image[k - s, i - s, j - s] >= image[k, i, j]:
                    binary_code[0] = 1
                if image[k - s, i - s, j ] >= image[k, i, j]:
                    binary_code[1] = 1
                if image[k - s, i - s, j + s] >= image[k, i, j]:
                    binary_code[2] = 1
                if image[k - s, i, j + s] >= image[k, i, j]:
                    binary_code[3] = 1
                if image[k - s, i + s, j + s] >= image[k, i, j]:
                    binary_code[4] = 1
                if image[k - s, i + s, j] >= image[k, i, j]:
                    binary_code[5] = 1
                if image[k - s, i + s, j - s] >= image[k, i, j]:
                    binary_code[6] = 1
                if image[k - s, i, j - s] >= image[k, i, j]:
                    binary_code[7] = 1
                if image[k - s, i, j] >= image[k, i, j]:
                    binary_code[8] = 1
                if image[k, i - s, j - s] >= image[k, i, j]:
                    binary_code[9] = 1
                if image[k, i - s, j ] >= image[k, i, j]:
                    binary_code[10] = 1
                if image[k, i - s, j + s] >= image[k, i, j]:
                    binary_code[11] = 1
                if image[k, i, j + s] >= image[k, i, j]:
                    binary_code[12] = 1
                if image[k, i + s, j + s] >= image[k, i, j]:
                    binary_code[13] = 1
                if image[k, i + s, j] >= image[k, i, j]:
                    binary_code[14] = 1
                if image[k, i + s, j - s] >= image[k, i, j]:
                    binary_code[15] = 1
                if image[k, i, j - s] >= image[k, i, j]:
                    binary_code[16] = 1
                if image[k + s, i - s, j - s] >= image[k, i, j]:
                    binary_code[17] = 1
                if image[k + s, i - s, j ] >= image[k, i, j]:
                    binary_code[18] = 1
                if image[k + s, i - s, j + s] >= image[k, i, j]:
                    binary_code[19] = 1
                if image[k + s, i, j + s] >= image[k, i, j]:
                    binary_code[20] = 1
                if image[k + s, i + s, j + s] >= image[k, i, j]:
                    binary_code[21] = 1
                if image[k + s, i + s, j] >= image[k, i, j]:
                    binary_code[22] = 1
                if image[k + s, i + s, j - s] >= image[k, i, j]:
                    binary_code[23] = 1
                if image[k + s, i, j - s] >= image[k, i, j]:
                    binary_code[24] = 1
                if image[k + s, i, j] >= image[k, i, j]:
                    binary_code[25] = 1
                texture_matrix[k, i, j] = sum(binary_code * weight)
    return texture_matrix

def single_sample():

    image_feature = []  # image features 46 fractures and 127 non-fractures
    clinical_feature = []  # clinical features + BMD
    label = []

    main_dir = "E:/Experiment/Osteoporosis/Data/Classification/"
    label_path = "E:/Experiment/Osteoporosis/Data/Classification/Label.xlsx"
    clinical_path = "E:/Experiment/Osteoporosis/Data/Classification/HCS.xlsx"  # clinical data file
    df2 = pd.read_excel(io=label_path)
    df_clinical = pd.read_excel(io=clinical_path)
    files = [file for file in os.listdir(main_dir) if file.endswith(".h5")]  # image file folder
    PI = 0  # participant ID
    PI_list = []  # participant ID list

    for name in files:
        file = os.path.join(main_dir, name)
        file = h5py.File(file, 'r')
        image = np.array(file["t0"]['channel0'][:])  # load the image

        filename = os.path.splitext(name)[0]
        number = df2.loc[df2['File'] == filename, f'Number'].iloc[0]  # image file ID -- Participant ID

        # clinical data of participant
        age = df_clinical.loc[df_clinical['abserno'] == number, f'hbsage'].iloc[0]
        sex = df_clinical.loc[df_clinical['abserno'] == number, f'absex'].iloc[0]
        height = df_clinical.loc[df_clinical['abserno'] == number, f'hbsht'].iloc[0]
        weight = df_clinical.loc[df_clinical['abserno'] == number, f'hbswt'].iloc[0]
        bmi = df_clinical.loc[df_clinical['abserno'] == number, f'hbsbmi'].iloc[0]
        dcalcium = df_clinical.loc[df_clinical['abserno'] == number, f'dcalcium'].iloc[0]
        hbsfnbmd = df_clinical.loc[df_clinical['abserno'] == number, f'hbsfnbmd'].iloc[0]  # BMD value
        if sex == "Male":
            feature_sex = 0
        else:
            feature_sex = 1
        Tibia = df2.loc[df2['File'] == filename, f'Tibia'].iloc[0]  # only analyze the tibial CT scans

        # load the label: yes-fracture; no-non fracture; null
        anyfrac = df_clinical.loc[df_clinical['abserno'] == number, f'anyfrac'].iloc[0]
        if anyfrac == "Yes" and Tibia == 1:
            gt_classes = 1  # fracture
        elif anyfrac == "No" and Tibia == 1:
            gt_classes = 0  # non-fracture
        else:
            gt_classes = 2  # null

        if gt_classes == 0 or gt_classes == 1:
            label.append(gt_classes)
            # calculate the histogram
            lbp = lbp_3D(image, s=1)  # calculate the texture feature matrix
            max_bins = 352
            lbp = lbp.flatten()  # vectorization
            hist, bins = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
            image_feature.append(hist)
            clinical_feature.append([age, height, weight, bmi, dcalcium, feature_sex, hbsfnbmd])
            PI = PI + 1
            PI_list.append(PI)

        print(f'process {name}')

    image_feature = np.array(image_feature)
    clinical_feature = np.array(clinical_feature)

    # normalize the clinical feature matrix
    clinical_feature[np.isnan(clinical_feature)] = 0
    clinical_feature = normalize(clinical_feature, axis=0, norm='max')  # normalize each col of the clinical feature with max-min normalization

    data = np.hstack((image_feature, clinical_feature))  # Concatenate the image feature and clinical feature
    PI_list = np.array(PI_list).reshape([PI, 1])
    label = np.array(label).reshape([PI, 1])
    data = np.hstack((PI_list, data))
    data = np.hstack((data, label))  # 0:1 Participant ID; 1:72 derived features; 72:73 label
    np.savetxt("Data\data_participant_level.csv", data, delimiter=",")

def accuracy(test_label, score):

    predict = score[:,1]
    score = score-0.1
    for i in range(len(test_label)):
        if score[i, 1]>=0.5:
            predict[i] = 1
        else:
            predict[i] = 0
    acc = sum(test_label==predict)/len(test_label)
    auc = metrics.roc_auc_score(test_label, score[:, 1])
    return acc, auc, score

def image_clinical_bmd():

    main_dir = "E:/Experiment/Osteoporosis/Data/Classification/"
    label_path = "E:/Experiment/Osteoporosis/Data/Classification/Label.xlsx"  # participant ID -- image data files
    clinical_path = "E:/Experiment/Osteoporosis/Data/Classification/HCS.xlsx"  # clinical data file
    df2 = pd.read_excel(io=label_path)
    df_clinical = pd.read_excel(io=clinical_path)
    files = [file for file in os.listdir(main_dir) if file.endswith(".h5")]  # image file folder
    PI = 0  # participant ID
    PI_list = []  # participant ID list
    sample_number = 0
    image_feature = []  # image features  46 fractures and 127 non-fractures 46*6+127*2
    clinical_feature = []  # clinical features + BMD
    label = []

    folder = 'Outcome/' # create model output file folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    for name in files:
        file = os.path.join(main_dir, name)
        file = h5py.File(file, 'r')
        image = np.array(file["t0"]['channel0'][:])  # load the image
        filename = os.path.splitext(name)[0]
        number = df2.loc[df2['File'] == filename, f'Number'].iloc[0]  # image file ID -- Participant ID

        # clinical data of participant
        age = df_clinical.loc[df_clinical['abserno'] == number, f'hbsage'].iloc[0]
        sex = df_clinical.loc[df_clinical['abserno'] == number, f'absex'].iloc[0]
        height = df_clinical.loc[df_clinical['abserno'] == number, f'hbsht'].iloc[0]
        weight = df_clinical.loc[df_clinical['abserno'] == number, f'hbswt'].iloc[0]
        bmi = df_clinical.loc[df_clinical['abserno'] == number, f'hbsbmi'].iloc[0]
        dcalcium = df_clinical.loc[df_clinical['abserno'] == number, f'dcalcium'].iloc[0]
        hbsfnbmd = df_clinical.loc[df_clinical['abserno'] == number, f'hbsfnbmd'].iloc[0]  # BMD value
        if sex == "Male":
            feature_sex = 0
        else:
            feature_sex = 1
        Tibia = df2.loc[df2['File'] == filename, f'Tibia'].iloc[0]  # only analyze the tibial CT scans

        # load the label: yes-fracture; no-non fracture; null
        anyfrac = df_clinical.loc[df_clinical['abserno'] == number, f'anyfrac'].iloc[0]
        if anyfrac == "Yes" and Tibia == 1:
            gt_classes = 1  # fracture
        elif anyfrac == "No" and Tibia == 1:
            gt_classes = 0  # non-fracture
        else:
            gt_classes = 2  # null

        if gt_classes == 0:
            PI = PI + 1
            for k in (76, 93):
                label.append(gt_classes)
                image1 = image[k:k + 17]
                lbp = lbp_3D(image1, s=1)  # calculate the texture feature matrix
                # calculate the histogram
                max_bins = 352
                lbp = lbp.flatten()  # vectorization
                hist, bins = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
                image_feature.append(hist)
                clinical_feature.append([age, height, weight, bmi, dcalcium, feature_sex, hbsfnbmd])
                PI_list.append(PI)
                sample_number = sample_number + 1

        elif gt_classes == 1:
            PI = PI + 1
            for k in (8, 25, 42, 59, 76, 93):
                label.append(gt_classes)
                image1 = image[k:k + 17]
                lbp = lbp_3D(image1, s=1)  # calculate the texture feature matrix
                # calculate the histogram
                max_bins = 352
                lbp = lbp.flatten()  # vectorization
                hist, bins = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
                image_feature.append(hist)
                clinical_feature.append([age, height, weight, bmi, dcalcium, feature_sex, hbsfnbmd])
                PI_list.append(PI)
                sample_number = sample_number + 1

        print(f'process {name}')

    image_feature = np.array(image_feature)
    clinical_feature = np.array(clinical_feature)

    # normalize the clinical feature matrix
    clinical_feature[np.isnan(clinical_feature)] = 0
    clinical_feature = normalize(clinical_feature, axis=0, norm='max')  # normalize each col of the clinical feature with max-min normalization

    data = np.hstack((image_feature, clinical_feature))  # Concatenate the image feature and clinical feature
    PI_list = np.array(PI_list).reshape([sample_number, 1])
    label = np.array(label).reshape([sample_number, 1])
    data = np.hstack((PI_list, data))
    data = np.hstack((data, label))  # 0:1 Participant ID; 1:72 derived features; 72:73 label
    np.savetxt("Data/data_tibia.csv", data, delimiter=",")

    #data = np.loadtxt(open("Data/data_tibia.csv", "rb"), delimiter=",", skiprows=0)

    # cross validation  divide data into five subsets based on participant level
    number = 173
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    data5 = []
    for i in range(data.shape[0]):
        if data[i, 0] <= 35:
            data1.append(data[i])
        elif data[i, 0] > 35 and data[i, 0] <= 70:
            data2.append(data[i])
        elif data[i, 0] > 70 and data[i, 0] <= 105:
            data3.append(data[i])
        elif data[i, 0] > 105 and data[i, 0] <= 140:
            data4.append(data[i])
        elif data[i, 0] > 140:
            data5.append(data[i])
    data1 = np.array(data1)
    data2 = np.array(data2)
    data3 = np.array(data3)
    data4 = np.array(data4)
    data5 = np.array(data5)

    sample_output = []  # participant ID, fracture probability, label
    m = 1  # folder
    for dataset in (data1, data2, data3, data4, data5):
        # select one subset as the testing set
        test_data = dataset[:, 1:360]
        test_label = dataset[:, -1]
        # other subsets are used as training set
        dataset_list = [data1, data2, data3, data4, data5]
        train_data_whole = []
        for i in range(5):
            if (dataset.shape[0] != dataset_list[i].shape[0]):
                train_data_whole.extend(dataset_list[i])
            else:
                if (dataset == dataset_list[i]).all():
                    continue
                else:
                    train_data_whole.extend(dataset_list[i])
        train_data_whole = np.array(train_data_whole)
        train_data = train_data_whole[:, 1:360]  # 0:1 Participant ID; 1:360 derived image-clinical-bmd features; 360:361 label
        train_label = train_data_whole[:, -1]

        # train the random forest classifier
        model = RandomForestClassifier(random_state=50, n_estimators=100)
        model.fit(train_data, train_label)
        #predictions = model.predict(test_data)
        #print(metrics.classification_report(test_label, predictions))  # print the classification results
        score = model.predict_proba(test_data)
        acc, auc, score = accuracy(test_label, score)
        print(f"Folder{m}, AUC and Accuracy in sample level: ", auc, acc)
        for i in range(dataset.shape[0]):
            sample_output.append([dataset[i, 0], score[i, 1], dataset[i, -1]])
        m = m + 1

    sample_output = np.array(sample_output)
    output = np.zeros([number, 3])
    # select the sample with the maximum fracture probability as the predicted probability for the patient
    for i in range(number):  # participant
        output[i][0] = i + 1  # participant ID
        probability_list = []
        for j in range(sample_output.shape[0]):  # sample
            if output[i][0] == sample_output[j][0]:
                probability_list.append(sample_output[j][1])
                output[i][2] = sample_output[j][2]  # label
        output[i][1] = max(probability_list)  # predicted fracture probability of a participant
    print("AUC in participant level from cross validation: ", metrics.roc_auc_score(output[:, 2], output[:, 1]))
    np.savetxt("Outcome/outcome_CT_clinical_BMD.csv", output, delimiter=",")

def image_clinical():

    main_dir = "E:/Experiment/Osteoporosis/Data/Classification/"
    label_path = "E:/Experiment/Osteoporosis/Data/Classification/Label.xlsx"  # participant ID -- image data files
    clinical_path = "E:/Experiment/Osteoporosis/Data/Classification/HCS.xlsx"  # clinical data file
    df2 = pd.read_excel(io=label_path)
    df_clinical = pd.read_excel(io=clinical_path)
    files = [file for file in os.listdir(main_dir) if file.endswith(".h5")]  # image file folder
    PI = 0  # participant ID
    PI_list = []  # participant ID list
    sample_number = 0
    image_feature = []  # image features  46 fractures and 127 non-fractures 46*6+127*2
    clinical_feature = []  # clinical features + BMD
    label = []

    folder = 'Outcome/' # create model output file folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    for name in files:
        file = os.path.join(main_dir, name)
        file = h5py.File(file, 'r')
        image = np.array(file["t0"]['channel0'][:])  # load the image
        filename = os.path.splitext(name)[0]
        number = df2.loc[df2['File'] == filename, f'Number'].iloc[0]  # image file ID -- Participant ID

        # clinical data of participant
        age = df_clinical.loc[df_clinical['abserno'] == number, f'hbsage'].iloc[0]
        sex = df_clinical.loc[df_clinical['abserno'] == number, f'absex'].iloc[0]
        height = df_clinical.loc[df_clinical['abserno'] == number, f'hbsht'].iloc[0]
        weight = df_clinical.loc[df_clinical['abserno'] == number, f'hbswt'].iloc[0]
        bmi = df_clinical.loc[df_clinical['abserno'] == number, f'hbsbmi'].iloc[0]
        dcalcium = df_clinical.loc[df_clinical['abserno'] == number, f'dcalcium'].iloc[0]
        hbsfnbmd = df_clinical.loc[df_clinical['abserno'] == number, f'hbsfnbmd'].iloc[0]  # BMD value
        if sex == "Male":
            feature_sex = 0
        else:
            feature_sex = 1
        Tibia = df2.loc[df2['File'] == filename, f'Tibia'].iloc[0]  # only analyze the tibial CT scans

        # load the label: yes-fracture; no-non fracture; null
        anyfrac = df_clinical.loc[df_clinical['abserno'] == number, f'anyfrac'].iloc[0]
        if anyfrac == "Yes" and Tibia == 1:
            gt_classes = 1  # fracture
        elif anyfrac == "No" and Tibia == 1:
            gt_classes = 0  # non-fracture
        else:
            gt_classes = 2  # null

        if gt_classes == 0:
            PI = PI + 1
            for k in (76, 93):
                label.append(gt_classes)
                image1 = image[k:k + 17]
                lbp = lbp_3D(image1, s=1)  # calculate the texture feature matrix
                # calculate the histogram
                max_bins = 352
                lbp = lbp.flatten()  # vectorization
                hist, bins = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
                image_feature.append(hist)
                clinical_feature.append([age, height, weight, bmi, dcalcium, feature_sex, hbsfnbmd])
                PI_list.append(PI)
                sample_number = sample_number + 1

        elif gt_classes == 1:
            PI = PI + 1
            for k in (8, 25, 42, 59, 76, 93):
                label.append(gt_classes)
                image1 = image[k:k + 17]
                lbp = lbp_3D(image1, s=1)  # calculate the texture feature matrix
                # calculate the histogram
                max_bins = 352
                lbp = lbp.flatten()  # vectorization
                hist, bins = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
                image_feature.append(hist)
                clinical_feature.append([age, height, weight, bmi, dcalcium, feature_sex, hbsfnbmd])
                PI_list.append(PI)
                sample_number = sample_number + 1

        print(f'process {name}')

    image_feature = np.array(image_feature)
    clinical_feature = np.array(clinical_feature)

    # normalize the clinical feature matrix
    clinical_feature[np.isnan(clinical_feature)] = 0
    clinical_feature = normalize(clinical_feature, axis=0, norm='max')  # normalize each col of the clinical feature with max-min normalization

    data = np.hstack((image_feature, clinical_feature))  # Concatenate the image feature and clinical feature
    PI_list = np.array(PI_list).reshape([sample_number, 1])
    label = np.array(label).reshape([sample_number, 1])
    data = np.hstack((PI_list, data))
    data = np.hstack((data, label))  # 0:1 Participant ID; 1:72 derived features; 72:73 label
    np.savetxt("Data/data_tibia.csv", data, delimiter=",")

    #data = np.loadtxt(open("Data/data_tibia.csv", "rb"), delimiter=",", skiprows=0)

    # cross validation  divide data into five subsets based on participant level
    number = 173
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    data5 = []
    for i in range(data.shape[0]):
        if data[i, 0] <= 35:
            data1.append(data[i])
        elif data[i, 0] > 35 and data[i, 0] <= 70:
            data2.append(data[i])
        elif data[i, 0] > 70 and data[i, 0] <= 105:
            data3.append(data[i])
        elif data[i, 0] > 105 and data[i, 0] <= 140:
            data4.append(data[i])
        elif data[i, 0] > 140:
            data5.append(data[i])
    data1 = np.array(data1)
    data2 = np.array(data2)
    data3 = np.array(data3)
    data4 = np.array(data4)
    data5 = np.array(data5)

    sample_output = []  # participant ID, fracture probability, label
    m = 1  # folder
    for dataset in (data1, data2, data3, data4, data5):
        # select one subset as the testing set
        test_data = dataset[:, 1:359]   # 0:1 Participant ID; 1:359 derived image-clinical features; 360:361 label
        test_label = dataset[:, -1]
        # other subsets are used as training set
        dataset_list = [data1, data2, data3, data4, data5]
        train_data_whole = []
        for i in range(5):
            if (dataset.shape[0] != dataset_list[i].shape[0]):
                train_data_whole.extend(dataset_list[i])
            else:
                if (dataset == dataset_list[i]).all():
                    continue
                else:
                    train_data_whole.extend(dataset_list[i])
        train_data_whole = np.array(train_data_whole)
        train_data = train_data_whole[:, 1:359]  # 0:1 Participant ID; 1:359 derived image-clinical features; 360:361 label
        train_label = train_data_whole[:, -1]

        # train the random forest classifier
        model = RandomForestClassifier(random_state=50, n_estimators=100)
        model.fit(train_data, train_label)
        #predictions = model.predict(test_data)
        #print(metrics.classification_report(test_label, predictions))  # print the classification results
        score = model.predict_proba(test_data)
        acc, auc, score = accuracy(test_label, score)
        print(f"Folder{m}, AUC and Accuracy in sample level: ", auc, acc)
        for i in range(dataset.shape[0]):
            sample_output.append([dataset[i, 0], score[i, 1], dataset[i, -1]])
        m = m + 1

    sample_output = np.array(sample_output)
    output = np.zeros([number, 3])
    # select the sample with the maximum fracture probability as the predicted probability for the patient
    for i in range(number):  # participant
        output[i][0] = i + 1  # participant ID
        probability_list = []
        for j in range(sample_output.shape[0]):  # sample
            if output[i][0] == sample_output[j][0]:
                probability_list.append(sample_output[j][1])
                output[i][2] = sample_output[j][2]  # label
        output[i][1] = max(probability_list)  # predicted fracture probability of a participant
    print("AUC in participant level from cross validation: ", metrics.roc_auc_score(output[:, 2], output[:, 1]))
    np.savetxt("Outcome/outcome_CT_clinical.csv", output, delimiter=",")

def image():

    main_dir = "E:/Experiment/Osteoporosis/Data/Classification/"
    label_path = "E:/Experiment/Osteoporosis/Data/Classification/Label.xlsx"  # participant ID -- image data files
    clinical_path = "E:/Experiment/Osteoporosis/Data/Classification/HCS.xlsx"  # clinical data file
    df2 = pd.read_excel(io=label_path)
    df_clinical = pd.read_excel(io=clinical_path)
    files = [file for file in os.listdir(main_dir) if file.endswith(".h5")]  # image file folder
    PI = 0  # participant ID
    PI_list = []  # participant ID list
    sample_number = 0
    image_feature = []  # image features  46 fractures and 127 non-fractures 46*6+127*2
    clinical_feature = []  # clinical features + BMD
    label = []

    folder = 'Outcome/' # create model output file folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    for name in files:
        file = os.path.join(main_dir, name)
        file = h5py.File(file, 'r')
        image = np.array(file["t0"]['channel0'][:])  # load the image
        filename = os.path.splitext(name)[0]
        number = df2.loc[df2['File'] == filename, f'Number'].iloc[0]  # image file ID -- Participant ID

        # clinical data of participant
        age = df_clinical.loc[df_clinical['abserno'] == number, f'hbsage'].iloc[0]
        sex = df_clinical.loc[df_clinical['abserno'] == number, f'absex'].iloc[0]
        height = df_clinical.loc[df_clinical['abserno'] == number, f'hbsht'].iloc[0]
        weight = df_clinical.loc[df_clinical['abserno'] == number, f'hbswt'].iloc[0]
        bmi = df_clinical.loc[df_clinical['abserno'] == number, f'hbsbmi'].iloc[0]
        dcalcium = df_clinical.loc[df_clinical['abserno'] == number, f'dcalcium'].iloc[0]
        hbsfnbmd = df_clinical.loc[df_clinical['abserno'] == number, f'hbsfnbmd'].iloc[0]  # BMD value
        if sex == "Male":
            feature_sex = 0
        else:
            feature_sex = 1
        Tibia = df2.loc[df2['File'] == filename, f'Tibia'].iloc[0]  # only analyze the tibial CT scans

        # load the label: yes-fracture; no-non fracture; null
        anyfrac = df_clinical.loc[df_clinical['abserno'] == number, f'anyfrac'].iloc[0]
        if anyfrac == "Yes" and Tibia == 1:
            gt_classes = 1  # fracture
        elif anyfrac == "No" and Tibia == 1:
            gt_classes = 0  # non-fracture
        else:
            gt_classes = 2  # null

        if gt_classes == 0:
            PI = PI + 1
            for k in (76, 93):
                label.append(gt_classes)
                image1 = image[k:k + 17]
                lbp = lbp_3D(image1, s=1)  # calculate the texture feature matrix
                # calculate the histogram
                max_bins = 352
                lbp = lbp.flatten()  # vectorization
                hist, bins = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
                image_feature.append(hist)
                clinical_feature.append([age, height, weight, bmi, dcalcium, feature_sex, hbsfnbmd])
                PI_list.append(PI)
                sample_number = sample_number + 1

        elif gt_classes == 1:
            PI = PI + 1
            for k in (8, 25, 42, 59, 76, 93):
                label.append(gt_classes)
                image1 = image[k:k + 17]
                lbp = lbp_3D(image1, s=1)   # calculate the texture feature matrix
                # calculate the histogram
                max_bins = 352
                lbp = lbp.flatten()  # vectorization
                hist, bins = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
                image_feature.append(hist)
                clinical_feature.append([age, height, weight, bmi, dcalcium, feature_sex, hbsfnbmd])
                PI_list.append(PI)
                sample_number = sample_number + 1

        print(f'process {name}')

    image_feature = np.array(image_feature)
    clinical_feature = np.array(clinical_feature)

    # normalize the clinical feature matrix
    clinical_feature[np.isnan(clinical_feature)] = 0
    clinical_feature = normalize(clinical_feature, axis=0, norm='max')  # normalize each col of the clinical feature with max-min normalization

    data = np.hstack((image_feature, clinical_feature))  # Concatenate the image feature and clinical feature
    PI_list = np.array(PI_list).reshape([sample_number, 1])
    label = np.array(label).reshape([sample_number, 1])
    data = np.hstack((PI_list, data))
    data = np.hstack((data, label))  # 0:1 Participant ID; 1:72 derived features; 72:73 label

    np.savetxt("Data/data_tibia.csv", data, delimiter=",")

    #data = np.loadtxt(open("Data/data_tibia.csv", "rb"), delimiter=",", skiprows=0)

    # cross validation  divide data into five subsets based on participant level
    number = 173
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    data5 = []
    for i in range(data.shape[0]):
        if data[i, 0] <= 35:
            data1.append(data[i])
        elif data[i, 0] > 35 and data[i, 0] <= 70:
            data2.append(data[i])
        elif data[i, 0] > 70 and data[i, 0] <= 105:
            data3.append(data[i])
        elif data[i, 0] > 105 and data[i, 0] <= 140:
            data4.append(data[i])
        elif data[i, 0] > 140:
            data5.append(data[i])
    data1 = np.array(data1)
    data2 = np.array(data2)
    data3 = np.array(data3)
    data4 = np.array(data4)
    data5 = np.array(data5)

    sample_output = []  # participant ID, fracture probability, label
    m = 1 # folder
    for dataset in (data1, data2, data3, data4, data5):
        # select one subset as the testing set
        test_data = dataset[:, 1:353]   # 0:1 Participant ID; 1:353 derived image features; 360:361 label
        test_label = dataset[:, -1]
        # other subsets are used as training set
        dataset_list = [data1, data2, data3, data4, data5]
        train_data_whole = []
        for i in range(5):
            if (dataset.shape[0] != dataset_list[i].shape[0]):
                train_data_whole.extend(dataset_list[i])
            else:
                if (dataset == dataset_list[i]).all():
                    continue
                else:
                    train_data_whole.extend(dataset_list[i])
        train_data_whole = np.array(train_data_whole)
        train_data = train_data_whole[:, 1:353]  # 0:1 Participant ID; 1:353 derived image features; 360:361 label
        train_label = train_data_whole[:, -1]

        # train the random forest classifier
        model = RandomForestClassifier(random_state=50, n_estimators=100)
        model.fit(train_data, train_label)
        #predictions = model.predict(test_data)
        #print(metrics.classification_report(test_label, predictions))  # print the classification results
        score = model.predict_proba(test_data)
        acc, auc, score = accuracy(test_label, score)
        print(f"Folder{m}, AUC and Accuracy in sample level: ", auc, acc)
        for i in range(dataset.shape[0]):
            sample_output.append([dataset[i, 0], score[i, 1], dataset[i, -1]])
        m = m + 1

    sample_output = np.array(sample_output)
    output = np.zeros([number, 3])
    # select the sample with the maximum fracture probability as the predicted probability for the patient
    for i in range(number):  # participant
        output[i][0] = i + 1  # participant ID
        probability_list = []
        for j in range(sample_output.shape[0]):  # sample
            if output[i][0] == sample_output[j][0]:
                probability_list.append(sample_output[j][1])
                output[i][2] = sample_output[j][2]  # label
        output[i][1] = max(probability_list)  # predicted fracture probability of a participant
    print("AUC in participant level from cross validation: ", metrics.roc_auc_score(output[:, 2], output[:, 1]))
    np.savetxt("Outcome/outcome_CT.csv", output, delimiter=",")

if __name__ == '__main__':

    image_clinical_bmd()
    #image_clinical()
    #image()
