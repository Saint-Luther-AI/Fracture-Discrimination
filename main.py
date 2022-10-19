import numpy as np
import pandas as pd
import os
import h5py
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier

def lbp_3D(image): # 3D texture feature extraction
    z, w, h = image.shape
    texture_matrix = np.zeros([z, w, h])
    for k in range(1, z - 1):
        for i in range(1, w - 1):
            for j in range(1, h - 1):
                lbp = 0
                if image[k - 1, i - 1, j - 1] >= image[k, i, j]:
                    lbp = lbp + 1
                if image[k - 1, i - 1, j ] >= image[k, i, j]:
                    lbp = lbp + 2
                if image[k - 1, i - 1, j + 1] >= image[k, i, j]:
                    lbp = lbp + 3
                if image[k - 1, i, j + 1] >= image[k, i, j]:
                    lbp = lbp + 4
                if image[k - 1, i + 1, j + 1] >= image[k, i, j]:
                    lbp = lbp + 5
                if image[k - 1, i + 1, j] >= image[k, i, j]:
                    lbp = lbp + 6
                if image[k - 1, i + 1, j - 1] >= image[k, i, j]:
                    lbp = lbp + 7
                if image[k - 1, i, j - 1] >= image[k, i, j]:
                    lbp = lbp + 8
                if image[k - 1, i, j] >= image[k, i, j]:
                    lbp = lbp + 9
                if image[k, i - 1, j - 1] >= image[k, i, j]:
                    lbp = lbp + 10
                if image[k, i - 1, j ] >= image[k, i, j]:
                    lbp = lbp + 11
                if image[k, i - 1, j + 1] >= image[k, i, j]:
                    lbp = lbp + 12
                if image[k, i, j + 1] >= image[k, i, j]:
                    lbp = lbp + 13
                if image[k, i + 1, j + 1] >= image[k, i, j]:
                    lbp = lbp + 14
                if image[k, i + 1, j] >= image[k, i, j]:
                    lbp = lbp + 15
                if image[k, i + 1, j - 1] >= image[k, i, j]:
                    lbp = lbp + 16
                if image[k, i, j - 1] >= image[k, i, j]:
                    lbp = lbp + 17
                if image[k + 1, i - 1, j - 1] >= image[k, i, j]:
                    lbp = lbp + 18
                if image[k + 1, i - 1, j ] >= image[k, i, j]:
                    lbp = lbp + 19
                if image[k + 1, i - 1, j + 1] >= image[k, i, j]:
                    lbp = lbp + 20
                if image[k + 1, i, j + 1] >= image[k, i, j]:
                    lbp = lbp + 21
                if image[k + 1, i + 1, j + 1] >= image[k, i, j]:
                    lbp = lbp + 22
                if image[k + 1, i + 1, j] >= image[k, i, j]:
                    lbp = lbp + 23
                if image[k + 1, i + 1, j - 1] >= image[k, i, j]:
                    lbp = lbp + 24
                if image[k + 1, i, j - 1] >= image[k, i, j]:
                    lbp = lbp + 25
                if image[k + 1, i, j] >= image[k, i, j]:
                    lbp = lbp + 26
                texture_matrix[k, i, j] = lbp
    return texture_matrix

def sensitivity_specificity():

    output_matrix = np.loadtxt(open("outcome_CT_clinical_BMD.csv", "rb"), delimiter=",", skiprows=0)
    threshold = 0.5

    TP = FN = TN = FP = 0
    for i in range(output_matrix.shape[0]): # output_matrix: participant ID, fracture probability, label
        score = output_matrix[i][1]
        label = output_matrix[i][-1]

        if score >= threshold:
            predict = 1
        else:
            predict = 0

        if predict == 1 and label == 1:
            TP = TP + 1
        if predict == 0 and label == 1:
            FN = FN + 1
        if predict == 0 and label == 0:
            TN = TN + 1
        if predict == 1 and label == 0:
            FP = FP + 1

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    print(sensitivity, specificity)

def ROC(): # calculate the set of (FPR, TPR) points for model outputs

    output_matrix = np.loadtxt(open("outcome_CT_clinical_BMD.csv", "rb"), delimiter=",", skiprows=0)
    print("shape:", output_matrix.shape)
    score = output_matrix[:, 1]
    label = output_matrix[:, -1]  # output_matrix: participant ID, fracture probability, label

    prob_pos = np.array(score).reshape([output_matrix.shape[0], 1])
    test_label = np.array(label).reshape([output_matrix.shape[0], 1])
    threshold = np.arange(0, 1, 0.01)
    fpr = []
    tpr = []
    for i in range(len(threshold)):
        TP = TN = FP = FN = 0
        for j in range(len(test_label)):
            if prob_pos[j] > threshold[i]:
                predict = 1
            else:
                predict = 0
            if predict == 1 and test_label[j] == 1:
                TP = TP + 1
            if predict == 0 and test_label[j] == 1:
                FN = FN + 1
            if predict == 0 and test_label[j] == 0:
                TN = TN + 1
            if predict == 1 and test_label[j] == 0:
                FP = FP + 1
        FPR = FP / (FP + TN)
        TPR = TP / (TP + FN)
        fpr.append(FPR)
        tpr.append(TPR)
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    set = np.vstack((fpr, tpr))

    np.savetxt("ROC_CT_clinical_BMD.csv", set, delimiter=",")

def clinical_factor():  # analyze the clinical risk factors

    clinical_path = "E:/Experiment/Osteoporosis/Data/Classification/HCS.xlsx"
    df_clinical = pd.read_excel(io=clinical_path)
    number_list = df_clinical["abserno"]

    clinical_feature = np.zeros([345, 7])
    label = np.zeros([345, 1])
    i = 0
    PI = 1  # participant ID
    PI_list = []  # participant ID list
    for number in number_list:
        anyfrac = df_clinical.loc[df_clinical['abserno'] == number, f'anyfrac'].iloc[0]
        if anyfrac == "Yes":  # fracture case
            gt_classes = 1
        elif anyfrac == "No":  # non-fracture case
            gt_classes = 0
        else:                  # null
            gt_classes = 2
        if gt_classes == 1 or gt_classes == 0:
            label[i] = gt_classes
            age = df_clinical.loc[df_clinical['abserno'] == number, f'hbsage'].iloc[0]
            sex = df_clinical.loc[df_clinical['abserno'] == number, f'absex'].iloc[0]
            height = df_clinical.loc[df_clinical['abserno'] == number, f'hbsht'].iloc[0]
            weight = df_clinical.loc[df_clinical['abserno'] == number, f'hbswt'].iloc[0]
            bmi = df_clinical.loc[df_clinical['abserno'] == number, f'hbsbmi'].iloc[0]
            dcalcium = df_clinical.loc[df_clinical['abserno'] == number, f'dcalcium'].iloc[0]
            hbsfnbmd = df_clinical.loc[df_clinical['abserno'] == number, f'hbsfnbmd'].iloc[0]
            if sex == "Male":
                feature_sex = 0
            else:
                feature_sex = 1

            clinical_feature[i] = np.array([age, height, weight, bmi, dcalcium, feature_sex, hbsfnbmd])
            i = i + 1
            PI_list.append(PI)
            PI = PI + 1

    clinical_feature[np.isnan(clinical_feature)] = 0
    clinical_feature_1 = clinical_feature[:, 0:5]
    clinical_feature_1 = normalize(clinical_feature_1, axis=0, norm='max')  # normalization each col of the clinical feature[age, height, weight, bmi]
    clinical_feature[:, 0:5] = clinical_feature_1
    label = label.reshape([345, 1])
    PI_list = np.array(PI_list)
    PI_list = PI_list.reshape([345, 1])
    data = np.hstack((PI_list, clinical_feature))
    data = np.hstack((data, label))  # 0:1 Participant ID; 1:8 derived clinical features; 8:9 label
    # np.savetxt("data_clinical_BMD.csv", data, delimiter=",")

    # five-folder validation
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    data5 = []
    for i in range(345):
        if data[i, 0] <= 69:
            data1.append(data[i])
        elif data[i, 0] > 69 and data[i, 0] <= 138:
            data2.append(data[i])
        elif data[i, 0] > 138 and data[i, 0] <= 207:
            data3.append(data[i])
        elif data[i, 0] > 207 and data[i, 0] <= 276:
            data4.append(data[i])
        elif data[i, 0] > 276:
            data5.append(data[i])
    data1 = np.array(data1)
    data2 = np.array(data2)
    data3 = np.array(data3)
    data4 = np.array(data4)
    data5 = np.array(data5)

    model_ouput = []  # model output: participant ID, fracture probability, label
    m = 0  # the index of testing subset
    for dataset in (data1, data2, data3, data4, data5):
        # select one subset as the testing set
        test_data = dataset[:, 1:8] # 0:1 Participant ID; 1:8 derived clinical features; 8:9 label
        test_label = dataset[:, -1]
        # other subsets are used as training set
        dataset_list = [data1, data2, data3, data4, data5]
        train_data_whole = []
        for i in range(5):
            if (dataset == dataset_list[i]).all():
                continue
            else:
                train_data_whole.extend(dataset_list[i])
        train_data_whole = np.array(train_data_whole)
        train_data = train_data_whole[:, 1:8]  # 0:1 Participant ID; 1:8 derived clinical features; 8:9 label
        train_label = train_data_whole[:, -1]

        # train the random forest classifier
        model = RandomForestClassifier(random_state=50, n_estimators=100)
        model.fit(train_data, train_label)
        predictions = model.predict(test_data)
        print(classification_report(test_label, predictions))  # print the classification results
        score = model.predict_proba(test_data)
        score = score[:, 1]  # fracture probabiltiy predicted from the model
        score = np.array(score)
        score = score.reshape([len(test_label), 1])
        for i in range(dataset.shape[0]):
            model_ouput.append([dataset[i,0], score[i,0], dataset[i,-1]])
    model_ouput = np.array(model_ouput)
    np.savetxt("Outcome/outcome_clinical_BMD.csv", model_ouput, delimiter=",")

if __name__ == '__main__':

    image_feature = np.zeros([484, 352])   # image features  46 fractures and 127 non-fractures
    clinical_feature = np.zeros([484, 7])  # clinical features + BMD
    label = []

    main_dir = "E:/Experiment/Osteoporosis/Data/Classification/"
    label_path = "E:/Experiment/Osteoporosis/Data/Classification/Label.xlsx"
    clinical_path = "E:/Experiment/Osteoporosis/Data/Classification/HCS.xlsx" # clinical data file
    df2 = pd.read_excel(io=label_path)
    df_clinical = pd.read_excel(io=clinical_path)
    files = [file for file in os.listdir(main_dir) if file.endswith(".h5")]  # image file folder
    PI = 1 # participant ID
    PI_list = [] # participant ID list
    i = 0 # sample ID

    for name in files:
        file = os.path.join(main_dir, name)
        file = h5py.File(file, 'r')
        image = np.array(file["t0"]['channel0'][:]) # load the image

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
        Tibia = df2.loc[df2['File'] == filename, f'Tibia'].iloc[0] # only analyze the tibial CT scans

        # load the label: yes-fracture; no-non fracture; null
        anyfrac = df_clinical.loc[df_clinical['abserno'] == number, f'anyfrac'].iloc[0]
        if anyfrac == "Yes" and Tibia ==1:
            gt_classes = 1  # fracture
        elif anyfrac == "No" and Tibia ==1:
            gt_classes = 0  # non-fracture
        else:
            gt_classes = 2  # null

        if gt_classes == 0:
            for k in (22, 44):
                label.append(gt_classes)
                image1 = image[k:k+22]
                lbp = lbp_3D(image1) # calculate the texture feature matrix
                # calculate the histogram
                hist = []
                max_bins = int(lbp.max() + 1)  # 352
                lbp = lbp.flatten() # vectorization
                hist, bins = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
                image_feature[i] = np.array(hist)
                clinical_feature[i] = np.array([age, height, weight, bmi, dcalcium, feature_sex, hbsfnbmd])
                PI_list.append(PI)
                i=i+1
            PI = PI + 1

        elif gt_classes == 1:
            for k in (0, 22, 44, 66, 88):
                label.append(gt_classes)
                image1 = image[k:k+22]
                lbp = lbp_3D(image1)
                # calculate the histogram
                hist = []
                max_bins = int(lbp.max() + 1)  # 352
                lbp = lbp.flatten() # vectorization
                hist, bins = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
                image_feature[i] = np.array(hist)
                clinical_feature[i] = np.array([age, height, weight, bmi, dcalcium, feature_sex, hbsfnbmd])
                PI_list.append(PI)
                i=i+1
            PI = PI + 1

        print(f'process {name}')

    # normalize the clinical feature matrix
    clinical_feature[np.isnan(clinical_feature)] = 0
    clinical_feature_1 = clinical_feature[:, 0:5]
    clinical_feature_1 = normalize(clinical_feature_1, axis=0, norm='max')
    clinical_feature[:, 0:5] = clinical_feature_1

    data = np.hstack((image_feature, clinical_feature))  # Concatenate the image feature and clinical feature
    PI_list = np.array(PI_list)
    PI_list = PI_list.reshape([484, 1])
    label = np.array(label)
    label = label.reshape([484, 1])
    data = np.hstack((PI_list, data))
    data = np.hstack((data, label))  # 0:1 Participant ID; 1:360 derived features; 361:362 label
    #np.savetxt("data.csv", data, delimiter=",")

    # five-folder validation
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    data5 = []
    for i in range(484):
        if data[i,0]<=35:
            data1.append(data[i])
        elif data[i,0]>35 and data[i,0]<=70:
            data2.append(data[i])
        elif data[i,0]>70 and data[i,0]<=105:
            data3.append(data[i])
        elif data[i,0]>105 and data[i,0]<=140:
            data4.append(data[i])
        elif data[i,0]>140:
            data5.append(data[i])
    data1 = np.array(data1)
    data2 = np.array(data2)
    data3 = np.array(data3)
    data4 = np.array(data4)
    data5 = np.array(data5)

    model_ouput = [] # model output: participant ID, fracture probability, label
    for dataset in (data1, data2, data3, data4, data5):

        # select one subset as the testing set
        test_data = dataset[:, 1:360]
        test_label = dataset[:, -1]
        # other subsets are used as training set
        dataset_list = [data1, data2, data3, data4, data5]
        train_data_whole = []
        for i in range(5):
            if (dataset == dataset_list[i]).all():
                continue
            else:
                train_data_whole.extend(dataset_list[i])
        train_data_whole = np.array(train_data_whole)
        train_data = train_data_whole[:, 1:360]  # 0:1 Participant ID; 1:360 derived features; 360:361 label
        train_label = train_data_whole[:, -1]

        # train the random forest classifier
        model = RandomForestClassifier(random_state=50, n_estimators=100)
        model.fit(train_data, train_label)
        predictions = model.predict(test_data)
        print(classification_report(test_label, predictions)) # print the classification results
        score = model.predict_proba(test_data)
        score = score[:, 1]  # fracture probabiltiy predicted from the model
        score = np.array(score)
        score = score.reshape([len(test_label), 1])

        # select the sample with the maximum fracture probability as the predicted probability for the patient
        i=0
        while i<(dataset.shape[0]-2):
            if dataset[i, -1] == 0: # non-fracture case
                max_probability = max(score[i], score[i+1])
                model_ouput.append([dataset[i,0], max_probability, dataset[i,-1]])
                i = i + 2
            if dataset[i, -1] == 1: # fracture case
                max_probability = max(score[i], score[i+1], score[i+2], score[i+3], score[i+4])
                model_ouput.append([dataset[i,0], max_probability, dataset[i,-1]])
                i = i + 5
    model_ouput=np.array(model_ouput)
    np.savetxt("Outcome/outcome_CT_clinical_BMD.csv", model_ouput, delimiter=",")
