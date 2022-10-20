import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

def plot_color():

    CT_clinical_BMD = np.loadtxt(open("Outcome/Tibia/outcome_CT_clinical_BMD.csv", "rb"), delimiter=",", skiprows=0)
    fpr, tpr, thresholds = metrics.roc_curve(CT_clinical_BMD[:, 2], CT_clinical_BMD[:, 1], pos_label=1)

    CT_clinical = np.loadtxt(open("Outcome/Tibia/outcome_CT_clinical.csv", "rb"), delimiter=",", skiprows=0)
    fpr2, tpr2, thresholds2 = metrics.roc_curve(CT_clinical[:, 2], CT_clinical[:, 1], pos_label=1)

    CT = np.loadtxt(open("Outcome/Tibia/outcome_CT.csv", "rb"), delimiter=",", skiprows=0)
    fpr3, tpr3, thresholds3 = metrics.roc_curve(CT[:, 2], CT[:, 1], pos_label=1)

    clinical_BMD = np.loadtxt(open("Outcome/Tibia/outcome_clinical_BMD.csv", "rb"), delimiter=",", skiprows=0)
    fpr4, tpr4, thresholds4 = metrics.roc_curve(clinical_BMD[:, 2], clinical_BMD[:, 1], pos_label=1)

    clinical = np.loadtxt(open("Outcome/Tibia/outcome_clinical.csv", "rb"), delimiter=",", skiprows=0)
    fpr5, tpr5, thresholds5 = metrics.roc_curve(clinical[:, 2], clinical[:, 1], pos_label=1)

    BMD = np.loadtxt(open("Outcome/Tibia/outcome_BMD.csv", "rb"), delimiter=",", skiprows=0)
    fpr6, tpr6, thresholds6 = metrics.roc_curve(BMD[:, 2], BMD[:, 1], pos_label=1)

    # plot the ROC curves
    plt.figure()
    plt.plot(fpr, tpr, label='HR-pQCT image data, clinical data and BMD', color='blue', linewidth=4)
    plt.plot(fpr2, tpr2, label='HR-pQCT image data and clinical data', color='orange', linewidth=2)
    plt.plot(fpr3, tpr3, label='HR-pQCT image data', color='green', linewidth=2)
    plt.plot(fpr4, tpr4, label='Clinical data and BMD', color='red', linewidth=2)
    plt.plot(fpr5, tpr5, label='Clinical data', color='purple', linewidth=2)
    plt.plot(fpr6, tpr6, label='BMD', color='brown', linewidth=2)

    # plot settings
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([-0.005,1.0])
    plt.ylim([0.0,1.01])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc='lower right')
    plt.show()

def plot_grey():

    CT_clinical_BMD = np.loadtxt(open("Outcome/Tibia/outcome_CT_clinical_BMD.csv", "rb"), delimiter=",", skiprows=0)
    fpr, tpr, thresholds = metrics.roc_curve(CT_clinical_BMD[:, 2], CT_clinical_BMD[:, 1], pos_label=1)

    clinical_BMD = np.loadtxt(open("Outcome/Tibia/outcome_clinical_BMD.csv", "rb"), delimiter=",", skiprows=0)
    fpr4, tpr4, thresholds4 = metrics.roc_curve(clinical_BMD[:, 2], clinical_BMD[:, 1], pos_label=1)

    BMD = np.loadtxt(open("Outcome/Tibia/outcome_BMD.csv", "rb"), delimiter=",", skiprows=0)
    fpr6, tpr6, thresholds6 = metrics.roc_curve(BMD[:, 2], BMD[:, 1], pos_label=1)

    # plot the ROC curves
    plt.figure()
    plt.plot(fpr, tpr, label='HR-pQCT image data, clinical data and BMD', color='Black', linewidth=4)

    plt.plot(fpr4, tpr4, label='Clinical data and BMD', color='grey', linewidth=4)
    plt.plot(fpr6, tpr6, label='BMD', color='black', linewidth=2)

    # plot settings
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([-0.005,1.0])
    plt.ylim([0.0,1.01])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':

    #plot_color()
    plot_grey()