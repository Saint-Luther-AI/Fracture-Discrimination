import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def clinical_BMD():  # analyze the clinical risk factors and BMD

    clinical_path = "E:/Experiment/Osteoporosis/Data/Classification/HCS.xlsx"
    df_clinical = pd.read_excel(io=clinical_path)
    number_list = df_clinical["abserno"]
    folder = 'Outcome/' # create model output file folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    clinical_feature = []
    PI = 1  # participant ID
    for number in number_list:
        anyfrac = df_clinical.loc[df_clinical['abserno'] == number, f'anyfrac'].iloc[0]
        if anyfrac == "Yes":  # fracture case
            gt_classes = 1
        elif anyfrac == "No":  # non-fracture case
            gt_classes = 0
        else:                  # null
            gt_classes = 2
        if gt_classes == 1 or gt_classes == 0:
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
            clinical_feature.append([PI, age, height, weight, bmi, dcalcium, feature_sex, hbsfnbmd, gt_classes])
            PI = PI + 1

    clinical_feature = np.array(clinical_feature)
    clinical_feature[np.isnan(clinical_feature)] = 0
    clinical_feature_1 = clinical_feature[:, 1:8] # [age, height, weight, bmi, dcalcium, feature_sex, hbsfnbmd]
    clinical_feature_1 = normalize(clinical_feature_1, axis=0, norm='max')  # normalize each col of the clinical feature with max-min normalization
    clinical_feature[:, 1:8] = clinical_feature_1

    data = clinical_feature  # 0:1 Participant ID; 1:8 derived clinical features; 8:9 label
    np.savetxt("Data/data_clinical_BMD.csv", data, delimiter=",")

    #fracture prediction
    train_data, test_data, train_label, test_label = train_test_split(data[:, 1:8], data[:, -1], random_state=18, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(train_data, train_label)
    model_output = []  # model output: participant ID, fracture probability, label
    predictions = model.predict(test_data)
    print(metrics.classification_report(test_label, predictions))  # print the classification results
    score = model.predict_proba(test_data)  # [0:non-fracture probability, 1:fracture probabiltiy]
    print("AUC: ", metrics.roc_auc_score(test_label, score[:, 1]))
    for i in range(len(test_label)):
        model_output.append([i, score[i, 1], test_label[i]])
    model_output = np.array(model_output)
    np.savetxt("Outcome/outcome_clinical_BMD.csv", model_output, delimiter=",")

def clinical():

    clinical_path = "E:/Experiment/Osteoporosis/Data/Classification/HCS.xlsx"
    df_clinical = pd.read_excel(io=clinical_path)
    number_list = df_clinical["abserno"]
    folder = 'Outcome/'  # create model output file folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    clinical_feature = []
    PI = 1  # participant ID
    for number in number_list:
        anyfrac = df_clinical.loc[df_clinical['abserno'] == number, f'anyfrac'].iloc[0]
        if anyfrac == "Yes":  # fracture case
            gt_classes = 1
        elif anyfrac == "No":  # non-fracture case
            gt_classes = 0
        else:  # null
            gt_classes = 2
        if gt_classes == 1 or gt_classes == 0:
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
            clinical_feature.append([PI, age, height, weight, bmi, dcalcium, feature_sex, hbsfnbmd, gt_classes])
            PI = PI + 1

    clinical_feature = np.array(clinical_feature)
    clinical_feature[np.isnan(clinical_feature)] = 0
    clinical_feature_1 = clinical_feature[:, 1:8]  # [age, height, weight, bmi, dcalcium, feature_sex, hbsfnbmd]
    clinical_feature_1 = normalize(clinical_feature_1, axis=0, norm='max')  # normalize each col of the clinical feature with max-min normalization
    clinical_feature[:, 1:8] = clinical_feature_1

    data = clinical_feature  # 0:1 Participant ID; 1:8 derived clinical features; 8:9 label
    np.savetxt("Data/data_clinical_BMD.csv", data, delimiter=",")

    # fracture prediction
    train_data, test_data, train_label, test_label = train_test_split(data[:,1:7], data[:,-1], random_state=0, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(train_data, train_label)
    model_output = []
    predictions = model.predict(test_data)
    print(metrics.classification_report(test_label, predictions))  # print the classification results
    score = model.predict_proba(test_data)  # [0:non-fracture probability, 1:fracture probabiltiy]
    print("AUC: ", metrics.roc_auc_score(test_label, score[:, 1]))
    for i in range(len(test_label)):
        model_output.append([i, score[i, 1], test_label[i]])
    model_output = np.array(model_output)
    np.savetxt("Outcome/outcome_clinical.csv", model_output, delimiter=",")

def BMD():

    clinical_path = "E:/Experiment/Osteoporosis/Data/Classification/HCS.xlsx"
    df_clinical = pd.read_excel(io=clinical_path)
    number_list = df_clinical["abserno"]
    folder = 'Outcome/'  # create model output file folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    clinical_feature = []
    PI = 1  # participant ID
    for number in number_list:
        anyfrac = df_clinical.loc[df_clinical['abserno'] == number, f'anyfrac'].iloc[0]
        if anyfrac == "Yes":  # fracture case
            gt_classes = 1
        elif anyfrac == "No":  # non-fracture case
            gt_classes = 0
        else:  # null
            gt_classes = 2
        if gt_classes == 1 or gt_classes == 0:
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
            clinical_feature.append([PI, age, height, weight, bmi, dcalcium, feature_sex, hbsfnbmd, gt_classes])
            PI = PI + 1

    clinical_feature = np.array(clinical_feature)
    clinical_feature[np.isnan(clinical_feature)] = 0
    clinical_feature_1 = clinical_feature[:, 1:8]  # [age, height, weight, bmi, dcalcium, feature_sex, hbsfnbmd]
    clinical_feature_1 = normalize(clinical_feature_1, axis=0, norm='max')  # normalize each col of the clinical feature with max-min normalization
    clinical_feature[:, 1:8] = clinical_feature_1

    data = clinical_feature  # 0:1 Participant ID; 1:8 derived clinical features; 8:9 label
    np.savetxt("Data/data_clinical_BMD.csv", data, delimiter=",")

    # fracture prediction
    data = data[:, -2:] # [BMD,label]
    train_data, test_data, train_label, test_label = train_test_split(data[:,0:1], data[:,1:2], random_state=18, test_size=0.2)
    model = LogisticRegression()
    model.fit(train_data, train_label)
    model_output = []
    predictions = model.predict(test_data)
    print(metrics.classification_report(test_label, predictions))  # print the classification results
    score = model.predict_proba(test_data)  # [0:non-fracture probability, 1:fracture probabiltiy]
    print("AUC: ", metrics.roc_auc_score(test_label, score[:, 1]))
    for i in range(len(test_label)):
        model_output.append([i, score[i, 1], test_label[i,0]])
    model_output = np.array(model_output)
    np.savetxt("Outcome/outcome_BMD.csv", model_output, delimiter=",")

if __name__ == '__main__':

    clinical_BMD()
    #clinical()
    #BMD()
