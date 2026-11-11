import numpy as np
import time
from operator import truediv
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score
from . import config
import warnings
import torch

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

warnings.filterwarnings("ignore")

def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def getReport(datasetType, model_bundle, model_adapter, test_loader, report_path, class_names, device, model):
    net = model_bundle["net"]
    net.eval()
    predictions = []
    labels = []
    for values in test_loader:
        if len(values) == 3:
            hsi_pca, sar, test_labels = values
            batch = {
                "hsi_pca": hsi_pca.to(device, non_blocking=True),
                "aux": sar.to(device, non_blocking=True),
                "label": test_labels.to(device, non_blocking=True),
            }
        else:
            hsi_pca, hsi, sar, test_labels = values
            batch = {
                "hsi_pca": hsi_pca.to(device, non_blocking=True),
                "hsi": hsi.to(device, non_blocking=True),
                "aux": sar.to(device, non_blocking=True),
                "label": test_labels.to(device, non_blocking=True),
            }
        with torch.inference_mode():
            outputs = model_adapter.forward_eval(model_bundle, batch)

        predictions.append(outputs.argmax(dim=1).cpu().numpy())
        labels.append(test_labels.cpu().numpy())

    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(labels)

    labels = list(range(len(class_names)))
    classification = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    oa = accuracy_score(y_true, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = np.nan_to_num(cohen_kappa_score(y_true, y_pred), nan=0.0)
    classification = str(classification)
    confusion = str(confusion)
    oa = oa * 100
    each_acc = each_acc * 100
    aa = aa * 100
    kappa = kappa * 100
    print('{} Overall accuracy (%)'.format(oa))
    print('{} Average accuracy (%)'.format(aa))
    print('{} Kappa accuracy (%)'.format(kappa))
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    current_time_log = 'Report time: {}\n'.format(current_time)

    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, 'a+') as report:
        report.write('{}'.format(config.get_task_info()))
        report.write('\n')
        report.write('{}'.format(current_time_log))
        report.write('\n')
        report.write('{} Overall accuracy (%)'.format(oa))
        report.write('\n')
        report.write('{} Average accuracy (%)'.format(aa))
        report.write('\n')
        report.write('{} Kappa accuracy (%)'.format(kappa))
        report.write('\n\n')
        report.write('{}'.format(classification))
        report.write('\n')
        report.write('{}'.format(confusion))
        report.write('\n')

def getMyReport(datasetType, model_bundle, model_adapter, test_loader, report_path, device, model):
    class_names = {
        0: [
            "Healthy grass", "Stressed grass", "Synthetic grass", "Trees", "Soil",
            "Water", "Residential", "Commercial", "Road", "Highway", "Railway",
            "Parking lot 1", "Parking lot 2", "Tennis court", "Running track",
        ],
        1: [
            "Healthy grass", "Stressed grass", "Artificial turf", "Evergreen trees",
            "Deciduous trees", "Bare earth", "Water", "Residential buildings",
            "Non-residential buildings", "Roads", "Sidewalks", "Crosswalks",
            "Major thoroughfares", "Highways", "Railways", "Paved parking lots",
            "Unpaved parking lots", "Cars", "Trains", "Stadium seats",
        ],
        2: ["Apple trees", "Buildings", "Ground", "Woods", "Vineyard", "Roads"],
        3: [
            "Forest", "Residential Area", "Industrial Area", "Low Plants", "Soil",
            "Allotment", "Commercial Area", "Water",
        ],
        4: [
            "Forest", "Residential Area", "Industrial Area", "Low Plants",
            "Allotment", "Commercial Area", "Water",
        ],
    }
    if datasetType not in class_names:
        raise ValueError(f"Unsupported dataset_type: {datasetType}")
    getReport(
        datasetType,
        model_bundle,
        model_adapter,
        test_loader,
        report_path,
        class_names[datasetType],
        device,
        model,
    )
    print("Report Success!")
