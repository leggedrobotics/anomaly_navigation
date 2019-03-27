def computeMaxYoudensIndex(fpr, tpr, thr):
    max_val = 0
    max_ind = None
    for i, (fp, tp) in enumerate(zip(fpr, tpr)):
        cur_val = tp + (1-fp)
        if cur_val > max_val:
            max_val = cur_val
            max_ind = i
    print('FPR: ' + str(fpr[max_ind]))
    print('TPR: ' + str(tpr[max_ind]))
    print('Thr: ' + str(thr[max_ind]))
    return thr[max_ind]



def computeTprAt5Fpr(fpr, tpr):
    return computeTprAtXFpr(fpr, tpr, 0.05)



def computeTprAtXFpr(fpr, tpr, fpr_thres):
    crit_ind = 0
    for ind, (fp, tp) in enumerate(zip(fpr, tpr)):
        if fp > fpr_thres:
            crit_ind = ind
            break
    if crit_ind == 0:
        fp_lower = 0.0
        tp_lower = 0.0
    else:
        fp_lower = fpr[crit_ind-1]
        tp_lower = tpr[crit_ind-1]
    fp_upper = fpr[crit_ind]
    tp_upper = tpr[crit_ind]

    tp_crit = tp_lower + (tp_upper-tp_lower)*(fpr_thres-fp_lower)/(fp_upper-fp_lower)
    return tp_crit

