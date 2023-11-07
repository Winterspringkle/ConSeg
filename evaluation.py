import json

import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff


# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold  # [8,1,96,96]
    GT = GT == torch.max(GT)
    SR = SR.int()
    GT = GT.int()
    corr = torch.sum(SR == GT)

    corr = corr.int()

    tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3)
    acc = float(corr) / float(tensor_size)

    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    SR = SR.int()
    GT = GT.int()

    TP = ((SR == 1).int() + (GT == 1).int()) == 2
    FN = ((SR == 0).int() + (GT == 1).int()) == 2

    TP = TP.int()
    FN = FN.int()

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE  # recall


def get_specificity(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    SR = SR.int()
    GT = GT.int()
    # TN : True Negative
    # FP : False Positive
    TN = ((SR == 0).int() + (GT == 0).int()) == 2
    FP = ((SR == 1).int() + (GT == 0).int()) == 2

    TN = TN.int()
    FP = FP.int()
    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SP


def get_precision(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    SR = SR.int()
    GT = GT.int()
    # TP : True Positive
    # FP : False Positive
    TP = ((SR == 1).int() + (GT == 1).int()) == 2
    FP = ((SR == 1).int() + (GT == 0).int()) == 2
    TP = TP.int()
    FP = FP.int()
    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

    return PC


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    # print("SR", SR)
    # print("GT", GT)
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)
    # print("SE",SE)
    # print("PC", PC)
    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)

    SR = SR.int()
    GT = GT.int()

    Inter = torch.sum((SR + GT) == 2)
    Union = torch.sum((SR + GT) >= 1)  # (0,1)(1,0)(1,1)

    JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    SR = SR.int()
    GT = GT.int()
    Inter = torch.sum((SR + GT) == 2)
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

    return DC

# Iou
def get_IOU(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    SR = SR.int()
    GT = GT.int()
    Inter = torch.sum((SR + GT) == 2)
    IOU = float(Inter) / (float(torch.sum(SR) + torch.sum(GT))-float(Inter) + 1e-6)

    return IOU

####by kun wang
def get_HD(SR, GT, threshold=0.5):
    # datas = np.load("D:\pythonprgram\EANet-master\data_tool\\test.npz",allow_pickle=True)
    # f = open("D:\pythonprgram\EANet-master\data_tool\\lover.json", 'r')
    # content = f.read()
    # datas = json.loads(content)
    scale = [1.,1.]
    # scale = datas["1.3.6.1.4.1.14519.5.2.1.6279.6001.139713436241461669335487719526"]["numpySpacing"][:2]
    # print(scale)
    SR = SR > threshold
    SR = SR.squeeze().detach().cpu().numpy()
    GT = GT == torch.max(GT)
    GT = GT.squeeze().detach().cpu().numpy()
    A_set = np.argwhere(SR == True)
    B_set = np.argwhere(GT == True)
    res1 = max(directed_hausdorff(A_set, B_set)[0], directed_hausdorff(A_set, B_set)[0])
    # print("HD1",res)
    A_set = (np.array([1 / scale[0], 1 / scale[1]])) * np.argwhere(SR == True)
    B_set = (np.array([1 / scale[0], 1 / scale[1]])) * np.argwhere(GT == True)
    res2 = max(directed_hausdorff(A_set, B_set)[0], directed_hausdorff(A_set, B_set)[0])
    # print("HD2",res)
    return res1, res2