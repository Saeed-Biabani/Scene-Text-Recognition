from src.utils.transforms import Resize, Normalization
from src.utils.labelConverter import CTCLabelConverter
from src.nn.ocr_model import RecognizerNetwork
from src.utils.dataset import DataGenerator
from torchvision.transforms import Compose
import torch.nn.functional as nnf
from src.utils.metrics import *
import config as cfg
import torch
import tqdm

device = cfg.device

testds = DataGenerator(
    root = cfg.ds_path["test_ds"],
    transforms = Compose([
        Resize((cfg.img_h, cfg.img_w)),
        Normalization()
    ])
);

model = RecognizerNetwork(cfg).to(device)
model.load_state_dict(torch.load("ocr.pth"))
model.eval()

converter = CTCLabelConverter(cfg)

prec_list = []
recal_list = []
auc_list = []

gt_list = []
pred_list = []
conf_list = []

loop = tqdm.tqdm(testds)
for (img, gt) in loop:
    img = img.to(device)

    preds = model(img[None, ...])
    preds = nnf.softmax(preds, dim = 2)
    preds_size = torch.IntTensor([preds.size(1)] * 1)

    pred_max_prob, preds_index = preds.max(2)
    confidence_score = pred_max_prob.cumprod(dim=0)[-1]


    txt = converter.decode(preds_index, preds_size)[0]
    
    pred_list.append(txt)
    gt_list.append(gt)
    conf = pred_max_prob.flatten().cpu().detach().numpy()
    conf_list.append(conf)

    prec_list.append(calculate_precision(gt, txt))
    recal_list.append(calculate_recall(gt, txt))
    
    loop.set_postfix({
        "Recall" : np.mean(recal_list),
        "Precision" : np.mean(prec_list),
    })

all_true_labels, all_conf_scores = accumulate_roc_data(gt_list, pred_list, conf_list)
fpr, tpr, thresholds = roc_curve(all_true_labels, all_conf_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for OCR Model')
plt.legend(loc="lower right")
plt.savefig("ROC.png")