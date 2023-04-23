import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import torch.distributed as dist

def calc_metrics(args, loader, label, label_onehot, model, criterion):
    acc, softmax, correct, logit, conf_correct, conf_wrong, ece_correct, ece_wrong, nll_correct, nll_wrong = get_metric_values(args, loader, model, criterion)
    # aurc, eaurc
    aurc, eaurc = calc_aurc_eaurc(softmax, correct)
    # fpr, aupr
    auroc, aupr_success, aupr, fpr, tnr = calc_fpr_aupr(softmax, correct)
    # calibration measure ece , mce, rmsce
    ece = calc_ece(softmax, label, bins=15)
    # brier, nll
    nll, brier = calc_nll_brier(softmax, logit, label, label_onehot)
    return acc, auroc*100, aupr_success*100, aupr*100, fpr*100, tnr*100, aurc*1000, eaurc*1000, ece*100, nll*10, brier*100

# AURC, EAURC
def calc_aurc_eaurc(softmax, correct):
    softmax = np.array(softmax)
    correctness = np.array(correct)
    softmax_max = np.max(softmax, 1)

    sort_values = sorted(zip(softmax_max[:], correctness[:]), key=lambda x:x[0], reverse=True)
    sort_softmax_max, sort_correctness = zip(*sort_values)
    risk_li, coverage_li = coverage_risk(sort_softmax_max, sort_correctness)
    aurc, eaurc = aurc_eaurc(risk_li)

    return aurc, eaurc

# AUPR ERROR
def calc_fpr_aupr(softmax, correct):
    softmax = np.array(softmax)
    correctness = np.array(correct)
    softmax_max = np.max(softmax, 1)

    fpr, tpr, thresholds = metrics.roc_curve(correctness, softmax_max)
    auroc = metrics.auc(fpr, tpr)
    idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
    fpr_in_tpr_95 = fpr[idx_tpr_95]
    tnr_in_tpr_95 = 1 - fpr[np.argmax(tpr >= .95)]

    precision, recall, thresholds = metrics.precision_recall_curve(correctness, softmax_max)
    aupr_success = metrics.auc(recall, precision)
    aupr_err = metrics.average_precision_score(-1 * correctness + 1, -1 * softmax_max)

    print("AUROC {0:.2f}".format(auroc * 100))
    print('AUPR_Success {0:.2f}'.format(aupr_success * 100))
    print("AUPR_Error {0:.2f}".format(aupr_err*100))
    print('FPR@TPR95 {0:.2f}'.format(fpr_in_tpr_95*100))
    print('TNR@TPR95 {0:.2f}'.format(tnr_in_tpr_95 * 100))

    return auroc, aupr_success, aupr_err, fpr_in_tpr_95, tnr_in_tpr_95

# ECE
def calc_ece(softmax, label, bins=15):
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmax = torch.tensor(softmax)
    labels = torch.tensor(label)

    softmax_max, predictions = torch.max(softmax, 1)
    correctness = predictions.eq(labels.long())

    ece = torch.zeros(1)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = softmax_max.gt(bin_lower.item()) * softmax_max.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = correctness[in_bin].float().mean()
            avg_confidence_in_bin = softmax_max[in_bin].mean()

            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    print("ECE {0:.2f} ".format(ece.item()*100))

    return ece.item()

# NLL & Brier Score
def calc_nll_brier(softmax, logit, label, label_onehot):
    brier_score = np.mean(np.sum((softmax - label_onehot) ** 2, axis=1))

    logit = torch.tensor(logit, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.int)
    logsoftmax = torch.nn.LogSoftmax(dim=1)

    log_softmax = logsoftmax(logit)
    nll = calc_nll(log_softmax, label)

    print("NLL {0:.2f} ".format(nll.item()*10))
    print('Brier {0:.2f}'.format(brier_score*100))

    return nll.item(), brier_score

# Calc NLL
def calc_nll(log_softmax, label):
    out = torch.zeros_like(label, dtype=torch.float)
    for i in range(len(label)):
        out[i] = log_softmax[i][label[i]]

    return -out.sum()/len(out)

# Calc coverage, risk
def coverage_risk(confidence, correctness):
    risk_list = []
    coverage_list = []
    risk = 0
    for i in range(len(confidence)):
        coverage = (i + 1) / len(confidence)
        coverage_list.append(coverage)

        if correctness[i] == 0:
            risk += 1

        risk_list.append(risk / (i + 1))

    return risk_list, coverage_list

# Calc aurc, eaurc
def aurc_eaurc(risk_list):
    r = risk_list[-1]
    risk_coverage_curve_area = 0
    optimal_risk_area = r + (1 - r) * np.log(1 - r)
    for risk_value in risk_list:
        risk_coverage_curve_area += risk_value * (1 / len(risk_list))

    aurc = risk_coverage_curve_area
    eaurc = risk_coverage_curve_area - optimal_risk_area

    print("AURC {0:.2f}".format(aurc*1000))
    print("EAURC {0:.2f}".format(eaurc*1000))

    return aurc, eaurc

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

# Get softmax, logit
def get_metric_values(args, loader, model, criterion):
    model.eval()
    prefetcher = data_prefetcher(loader)
    input, target = prefetcher.next()
    with torch.no_grad():
        total_loss = 0.
        total_acc = 0.
        accuracy = 0.

        list_softmax = []
        list_correct = []
        list_logit = []

        logits_list = []
        labels_list = []
        conf = []
        correct = []
        i = 0
        # if args.method == 'CS-KD':
        #     for input, target, in loader:
        #         input = input.cuda()
        #         target = target.cuda()
        #         output = model(input)
        #         loss = criterion(output, target).cuda()
        #         logits_list.append(output.detach().cpu())
        #         labels_list.append(target.cpu())
        #         total_loss += loss.mean().item()
        #         pred = output.data.max(1, keepdim=True)[1]
        #         prob, _pred = F.softmax(output, dim=1).max(1)
        #         conf.append(prob.detach().cpu().view(-1).numpy())
        #         correct.append(_pred.cpu().eq(target.cpu().data.view_as(_pred)).numpy())
        #         total_acc += pred.eq(target.data.view_as(pred)).sum()
        #
        #         for i in output:
        #             list_logit.append(i.cpu().data.numpy())
        #
        #         list_softmax.extend(F.softmax(output).cpu().data.numpy())
        #
        #         for j in range(len(pred)):
        #             if pred[j] == target[j]:
        #                 accuracy += 1
        #                 cor = 1
        #             else:
        #                 cor = 0
        #             list_correct.append(cor)
        #     logits = torch.cat(logits_list)
        #     labels = torch.cat(labels_list)
        #     conf = np.concatenate(conf, axis=0)
        #     correct = np.concatenate(correct, axis=0)
        #     groud = np.ones_like(correct)
        #     conf_wrong = np.mean(conf[groud ^ correct])
        #     conf_correct = np.mean(conf[correct])
        #     ece_wrong = ece_criterion(logits[groud ^ correct], labels[groud ^ correct]).item()
        #     ece_correct = ece_criterion(logits[correct], labels[correct]).item()
        #
        #     nll_wrong = nll_criterion(logits[groud ^ correct], labels[groud ^ correct]).item()
        #     nll_correct = nll_criterion(logits[correct], labels[correct]).item()
        #
        # else:

        # for input, target, idx in loader:
        while input is not None:
            i += 1
            input = input.cuda()
            target = target.long().cuda()
            output = model(input)
            loss = criterion(output, target).cuda()
            logits_list.append(output.detach().cpu())
            labels_list.append(target.cpu())
            total_loss += loss.mean().item()
            pred = output.data.max(1, keepdim=True)[1]
            prob, _pred = F.softmax(output, dim=1).max(1)
            conf.append(prob.detach().cpu().view(-1).numpy())
            correct.append(_pred.cpu().eq(target.cpu().data.view_as(_pred)).numpy())
            total_acc += pred.eq(target.data.view_as(pred)).sum()

            for i in output:
                list_logit.append(i.cpu().data.numpy())

            list_softmax.extend(F.softmax(output).cpu().data.numpy())

            for j in range(len(pred)):
                if pred[j] == target[j]:
                    accuracy += 1
                    cor = 1
                else:
                    cor = 0
                list_correct.append(cor)
            input, target = prefetcher.next()

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        conf = np.concatenate(conf, axis=0)
        correct = np.concatenate(correct, axis=0)
        groud = np.ones_like(correct)
        conf_wrong = np.mean(conf[groud ^ correct])
        conf_correct = np.mean(conf[correct])
        ece_wrong = ece_criterion(logits[groud ^ correct], labels[groud ^ correct]).item()
        ece_correct = ece_criterion(logits[correct], labels[correct]).item()

        nll_wrong = nll_criterion(logits[groud ^ correct], labels[groud ^ correct]).item()
        nll_correct = nll_criterion(logits[correct], labels[correct]).item()

        total_loss /= len(loader)
        print(total_acc,  len(loader.dataset))
        if args.distributed:
            total_acc = reduce_tensor(args, total_acc)
        total_acc = 100. * total_acc.item() / len(loader.dataset)
        print(total_acc)

        print('Accuracy {:.2f}'.format(total_acc))

    return total_acc, list_softmax, list_correct, list_logit, conf_correct, conf_wrong, ece_correct, ece_wrong, nll_correct, nll_wrong


def reduce_tensor(args, tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt

class ECELoss(nn.Module):
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


ece_criterion = ECELoss().cuda()
nll_criterion = nn.CrossEntropyLoss().cuda()