import numpy as np

def sensitivity_specificity():

    output_matrix = np.loadtxt(open("Outcome/outcome_CT_clinical_BMD.csv", "rb"), delimiter=",", skiprows=0)
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
    print("sensitivity, specificity", sensitivity, specificity)

    '''
    # choose the point on the ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(output_matrix[:, 2], output_matrix[:, 1], pos_label=1)
    index = np.where(thresholds == point)
    print("sensitivity, specificity", tpr[index], 1 - fpr[index])
    '''

def ROC(): # calculate the set of (FPR, TPR) points for model outputs

    output_matrix = np.loadtxt(open("Outcome/outcome_CT_clinical_BMD.csv", "rb"), delimiter=",", skiprows=0)
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