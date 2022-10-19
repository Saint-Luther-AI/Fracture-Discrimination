import numpy as np
import pandas as pd
import os
import h5py
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier

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

    clinical_factor()
