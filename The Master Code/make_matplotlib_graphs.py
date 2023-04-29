from matplotlib import pyplot as plt
import pandas as pd
import tensorboard as tb
import numpy as np


EXPERIMENT_ID = 'fkobeNTdSRuK0hGA0QoMSA'

figure_number = 1

def retrieve_data(experiment_id):
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    return experiment.get_scalars()

def plot_data(df, metric='Train/Loss', title='Loss', xlabel='Epoch', ylabel='Loss', hold_on=False):
    global figure_number
    # keep only the the rows where the tag == metric
    df = df[df['tag'] == metric]

    values = df['value'].values
    fig = plt.figure(figure_number)
    # plot the values
    plt.plot(values, label=metric)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend()

    if not hold_on:
        figure_number += 1


def make_augmentation_graphs(test_path='augmentation_test'):
    #combine test_path with the path to the csv file mutual_info.csv
    df = pd.read_csv(test_path + '/mutual_info.csv',  names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    # get the values in the second coloum
    train_test_mu = df.iloc[1:, 1].values
    train_test_mu = [float(mu) for mu in train_test_mu]
    combinations = df.iloc[1:, 3:10].values
    combinations[0][0] = '0'

    new_combinations = []
    for combination in combinations:
        #combine the numbers in combination
        combination = [int(comb) for comb in combination if not pd.isnull(comb)]
        combination = [int(comb) if type(comb) == float else comb for comb in combination]
        combination = ''.join(str(combination))
        new_combinations.append(combination)

    # sort the values in train_test_mu from lowest to highest and sort the combinations accordingly
    # train_test_mu, new_combinations = zip(*sorted(zip(train_test_mu, new_combinations)))


    # make a bar chart mutual information for each of the combinations
    fig = plt.figure(figure_number)
    # plt.bar(new_combinations, train_test_mu)
    plt.bar(np.arange(0, len(new_combinations), 1), train_test_mu)
    plt.xlabel('Combinations')
    plt.ylabel('Mutual Information')
    plt.title('Mutual Information for each combination')
    # dont show the values on the x axis
    # plt.xticks(np.arange(0, len(new_combinations), 1))
    # plt.yticks(np.arange(0, 0.5, 0.1))



    # plot the values
    # plt.plot(values, label='Mutual Information')


        




if __name__ == '__main__':
    # data = retrieve_data(EXPERIMENT_ID)
    # plot_data(df=data, metric='Train/Loss', title='Loss', xlabel='Epoch', ylabel='Loss')
    # plot_data(df=data, metric='Train/Loss_box_reg', title='Loss_box_reg', xlabel='Epoch', ylabel='Loss Box Reggression')
    # # plot_data(df=data, metric='Train/Loss_classifier', title='Loss_classifier', xlabel='Epoch', ylabel='Loss Classifier')
    # # plot_data(df=data, metric='Train/Loss_objectness', title='Loss_objectness', xlabel='Epoch', ylabel='Loss Objectness')
    # # plot_data(df=data, metric='Train/Loss_rpn_box_reg', title='Loss_rpn_box_reg', xlabel='Epoch', ylabel='Loss RPN Box Reggression')

    # # Validation/mAP with hold_on=True
    # plot_data(df=data, metric='Validation/mAP', title='mAP', xlabel='Epoch', ylabel='mAP', hold_on=True)
    # plot_data(df=data, metric='Validation/mAP_50', title='mAP_50', xlabel='Epoch', ylabel='mAP_50', hold_on=True)
    # plot_data(df=data, metric='Validation/mAP_75', title='mAP_75', xlabel='Epoch', ylabel='mAP_75')

    make_augmentation_graphs()

    plt.show()

