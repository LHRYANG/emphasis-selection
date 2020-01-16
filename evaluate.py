import numpy as np
import sys
import os
import os.path


def average(lst):
    return sum(lst) / float(len(lst))


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def match_m(all_scores, all_labels):
    """
    This function computes match_m.
    :param all_scores: submission scores
    :param all_labels: ground_truth labels
    :return: match_m dict
    """
    #print("[LOG] computing Match_m . . .")
    top_m = [1, 2, 3, 4]
    match_ms = {}
    rank_ms = {}
    for m in top_m:
        #print("[LOG] computing m={} in match_m".format(m))
        intersects_lst = []
        rank_lst=[]
        # ****************** computing scores:
        score_lst = []
        for s in all_scores:
            # the length of sentence needs to be more than m:
            if len(s) <= m:
                continue
            s = np.array(s)
            ind_score = np.argsort(s)[-m:]
            score_lst.append(ind_score.tolist())
        # ****************** computing labels:
        label_lst = []
        for l in all_labels:
            # the length of sentence needs to be more than m:
            if len(l) <= m:
                continue
            # if label list contains several top values with the same amount we consider them all
            h = m
            if len(l) > h:
                while (l[np.argsort(l)[-h]] == l[np.argsort(l)[-(h + 1)]] and h < (len(l) - 1)):
                    h += 1
            l = np.array(l)
            ind_label = np.argsort(l)[-h:]
            label_lst.append(ind_label.tolist())

        for i in range(len(score_lst)):
            # computing the intersection between scores and ground_truth labels:
            intersect = intersection(score_lst[i], label_lst[i])
            rank = (np.array(score_lst[i][-m:]) == np.array(label_lst[i][-m:]))
            #print("score: ",score_lst[i])
            #print("label: ",label_lst[i])
            rank_lst.append(np.sum(rank)/float((min(m, len(score_lst[i])))))
            intersects_lst.append((len(intersect))/float((min(m, len(score_lst[i])))))
        # taking average of intersects for the current m:
        match_ms[m] = average(intersects_lst)
        rank_ms[m] = average(rank_lst)
    return match_ms,rank_ms


def read_results(filename):
    lines = read_lines(filename) + ['']
    e_freq_lst, e_freq_lsts = [], []

    for line in lines:
        if line:
            splitted = line.split("\t")
            e_freq = splitted[2]
            e_freq_lst.append(e_freq)

        elif e_freq_lst:
            e_freq_lsts.append(e_freq_lst)
            e_freq_lst = []
    return e_freq_lsts


def read_labels(filename):
    lines = read_lines(filename) + ['']
    e_freq_lst, e_freq_lsts = [], []

    for line in lines:
        if line:
            splitted = line.split("\t")
            e_freq = splitted[4]
            e_freq_lst.append(e_freq)

        elif e_freq_lst:
            e_freq_lsts.append(e_freq_lst)
            e_freq_lst = []
    return e_freq_lsts


def read_lines(filename):
    with open(filename, 'r',encoding='utf-8') as fp:
        lines = [line.strip() for line in fp]
    return lines


def eval(input, target):
    input_dir = input
    output_dir = target

    #print("[LOG] input_dir: ", input_dir)
    #print("[LOG] output_dir: ", output_dir)

    output_file = open("scores.txt", "w")

    all_score = read_results(input_dir)
    all_label = read_labels(output_dir)

    assert len(all_score) == len(all_label)
    for i in range(len(all_label)):
        assert len(all_label[i]) == len(all_score[i])

    matchm,rankms = match_m(all_score, all_label)
    print("[LOG] Match_m: ", matchm)
    #print("[LOG] computing RANKING score")

    sum_of_all_scores = 0
    for key, value in matchm.items():
        output_file.write("score" + str(key) + ":" + str(value))
        output_file.write("\n")
        sum_of_all_scores += value
    output_file.write("score:" + str(sum_of_all_scores / float(4)) + "\n")  # score for final "computed score"
    output_file.close()
    return matchm,rankms



