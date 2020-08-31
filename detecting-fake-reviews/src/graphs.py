import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import copy

def bow_graph():
    print('bow graph')
    path = 'results/bow_results.txt'
    type_dict = {
        'TRAIN': '',
        'TEST' : ''
    }
    model_dict = {'logistic_regression': copy.deepcopy(type_dict),
                  'naive_bayes': copy.deepcopy(type_dict),
                  'knearest_neighbors': copy.deepcopy(type_dict),
                  'decision_trees': copy.deepcopy(type_dict),
                  'random_forest': copy.deepcopy(type_dict)}
    data_dict = {'chinese': copy.deepcopy(model_dict),
                 'yelp': copy.deepcopy(model_dict),
                 'op_spam': copy.deepcopy(model_dict)}
    cur_data_set = ''
    cur_model = ''
    cur_type = ''
    for line in open(path):
        line = line.strip().split()
        if len(line) > 1:
            # print(line)
            if line[1] in ['chinese','yelp','op_spam']:
                # print(line)
                cur_data_set = line[1]
            if line[1] in ['logistic_regression', 'naive_bayes', 'knearest_neighbors', 'decision_trees', 'random_forest']:
                # print(line)
                cur_model = line[1]
            if line[0] in ['TRAIN','TEST']:
                # print(line)
                cur_type = line[0]
            # if line[0] in ['accuracy:']:
            if line[0] in ['AUC']:
                print(cur_data_set, cur_model, cur_type, line[2])
                data_dict[cur_data_set][cur_model][cur_type] = line[2]
    print(data_dict)
    return data_dict

def make_bow_graph(data_dict):


    N = 5
    labels = ['logistic_regression', 'naive_bayes', 'knearest_neighbors', 'decision_trees', 'random_forest']
    chinese_acc = []
    yelp_acc = []
    op_acc = []


    for k,v in data_dict.items():
        print(k)
        for k2,v2 in data_dict[k].items():
            print(k2, v2['TRAIN'])
            if k == 'chinese':
                chinese_acc.append(float(v2['TRAIN']))
            if k == 'yelp':
                yelp_acc.append(float(v2['TRAIN']))
            if k == 'op_spam':
                op_acc.append(float(v2['TRAIN']))

    ind = np.arange(N)
    width = 0.25

    r1 = np.arange(len(chinese_acc))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]

    # Make the plot
    # plt.bar(r1, chinese_acc, color='#7f6d5f', width=width, edgecolor='white', label='Hauyi')
    # plt.bar(r2, yelp_acc, color='#557f2d', width=width, edgecolor='white', label='Yelp')
    # plt.bar(r3, op_acc, color='#2d7f5e', width=width, edgecolor='white', label='Op_spam')

    plt.bar(r1, chinese_acc, color='red', width=width, edgecolor='white', label='Hauyi')
    plt.bar(r2, yelp_acc, color='blue', width=width, edgecolor='white', label='Yelp')
    plt.bar(r3, op_acc, color='green', width=width, edgecolor='white', label='Op_spam')

    # Add xticks on the middle of the group bars
    plt.xlabel('Classifier', fontweight='bold')
    plt.xticks([r + width for r in range(len(chinese_acc))], ['logistic_regression', 'naive_bayes', 'knearest_neighbors', 'decision_trees', 'random_forest'])

    # plt.xlabel('Size logN Graph')
    plt.ylabel('AUC', fontweight='bold')
    plt.title('AUC of Classifiers for Each Dataset on Train Data', fontweight='bold')

    # Create legend & Show graphic
    plt.legend()
    plt.show()

def NN_epoch():
    print('NN epoch')
    path = 'results/op_spamNN.txt'
    cur_data_set = ''
    cur_data_type = ''
    measurements = {
        'Loss:': [],
        'Acc:': []
    }
    data_type = {
        'Train': copy.deepcopy(measurements),
        'Val.' : copy.deepcopy(measurements)
    }
    data_dict = {
        'yelp': copy.deepcopy(data_type),
        'op_spam': copy.deepcopy(data_type)
    }
    for line in open(path):
        line = line.strip().split()

        if len(line) > 1:
            # print(line)
            if line[0] in ['op_spam', 'yelp']:
                # print(line)
                cur_data_set = line[0]
            if line[0] in ['Train', 'Val.']:
                # print(line)
                cur_data_type = line[0]
                loss = line[2]
                acc = line[6]
                # print(cur_data_set, cur_data_type, 'Loss', loss)
                # print(cur_data_set, cur_data_type, 'Acc', acc)
                data_dict[cur_data_set][cur_data_type]['Loss:'].append(loss)
                data_dict[cur_data_set][cur_data_type]['Acc:'].append(acc)
    return data_dict

def make_nn_graph(data_dict):
    print('make nn graph')
    yelp = data_dict['yelp']['Val.']['Loss:']
    yelp = [float(x[0:5]) for x in yelp]
    op_spam = data_dict['op_spam']['Val.']['Loss:']
    op_spam = [float(x[0:5]) for x in op_spam]
    yelp2 = data_dict['yelp']['Train']['Loss:']
    yelp2 = [float(x[0:5]) for x in yelp2]
    op_spam2 = data_dict['op_spam']['Train']['Loss:']
    op_spam2 = [float(x[0:5]) for x in op_spam2]
    print(yelp)
    names = ['1', '2', '3', '4', '5']
    plt.plot(names, yelp, linewidth=3, label = 'Yelp Val.')
    plt.plot(names, op_spam, linewidth=3, label = 'Op_spam Val.')
    plt.plot(names, yelp2, linewidth=3, label = 'Yelp Train')
    plt.plot(names, op_spam2, linewidth=3, label = 'Op_spam Train')
    plt.ylim(0,1)
    plt.title('Loss of Training and Validation Data for NN over 5 Epochs', fontweight='bold')
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.legend()
    plt.show()


    op_spam = []


def docGraph():
    print('doc graph')
    # accuracy
    # yelp = [96.06, 96.24]
    # op_spam = [58.83, 56.00]
    groups = ['Training', 'Testing']

    #auc
    yelp = [72.28, 67.15]
    op_spam = [62.16, 61.82]

    N = 2
    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, yelp, width, label='Yelp')
    plt.bar(ind + width, op_spam, width,
            label='Op_spam')

    plt.ylabel('AUC', fontweight='bold')
    plt.title('Doc2Vec AUC by Dataset and Type of Data', fontweight='bold')

    plt.xticks(ind + width / 2, groups, fontweight='bold')
    plt.legend(loc='best')
    plt.show()



if __name__ == '__main__':

    print('graphs.py')

    # data_dict = bow_graph()
    # make_bow_graph(data_dict)
    # make_bow_graph(data_dict)

    # data_dict = NN_epoch()
    # make_nn_graph(data_dict)
    docGraph()
