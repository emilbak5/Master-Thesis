from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter

from copy import copy

from dataclasses import dataclass

import pandas as pd
import tensorboard as tb
import numpy as np

from tqdm import tqdm


EXPERIMENT_ID_RETINA = 'RgcBqiodQrqqedJIxkw6GQ'
EXPERIMENT_ID_RETINA_AUG = 'l3aR9kKQTPGuaQLlNGlBMw'

EXPERIMENT_ID_HP_TUNING_1 = 'BZRMoJNUQQ21h3MzTBxuGQ' # ssdlite
EXPERIMENT_ID_HP_TUNING_2 = 'eprN57ZDQyCYxjB3aCJN3Q' # ssd300
EXPERIMENT_ID_HP_TUNING_3 = 'seoS2uODRXecR2rDUduZcA' # fasterrcnn
EXPERIMENT_ID_HP_TUNING_4 = 'XQVxBr8YTqCvOD5RYXztkQ' # fasterrcnn_v2
EXPERIMENT_ID_HP_TUNING_5 = 'fzVL7EfmS0e8kQXe31NgeA' # retinanet
EXPERIMENT_ID_HP_TUNING_6 = 'ogIWV8lmQ9aA1ZxLBJamKQ' # retinanet_v2

@dataclass
class ExperimentNumbers:
    map = []
    epoch = []



figure_number = 1

def test(experiment_id, name):
    global figure_number
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()
    # only keep values where the tag is 'Validation/mAP'
    df = df[df['tag'] == 'Validation/mAP']
    values = list(df.values)


    all_runs = []
    prev_name = values[0][0]
    temp_numbers = ExperimentNumbers()
    temp_numbers.map = []
    temp_numbers.epoch = []

    first_value = True
    for value in values:

        if value[2] != 0 or first_value == True:
            temp_numbers.epoch.append(value[2])
            temp_numbers.map.append(value[3])
            prev_name = value[0]
            first_value = False
            if len(temp_numbers.epoch) > 20:
                x = 5
        else:
            all_runs.append(copy(temp_numbers))
            temp_numbers = ExperimentNumbers()
            temp_numbers.map = []
            temp_numbers.epoch = []
            temp_numbers.epoch.append(value[2])
            temp_numbers.map.append(value[3])
            prev_name = value[0]
    
    all_runs.append(copy(temp_numbers))

    fig = plt.figure(figure_number)

    longest_epoch = 0
    for run in tqdm(all_runs):
        epochs = np.arange(0, len(run.epoch))
        if len(run.epoch) > longest_epoch:
            longest_epoch = len(run.epoch)
        plt.plot(epochs , run.map)
        # add a small dot at the end of each graph that has the same color as the graph
        plt.plot(epochs[-1], run.map[-1], 'o', color=plt.gca().lines[-1].get_color(), markersize=3)
    
    plt.xticks(np.arange(0, longest_epoch + 1, 1.0))
    # set the minimumn y value to 0
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title(name)

    figure_number += 1

        


    x = 5
def retrieve_data(experiment_id):
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    return experiment.get_scalars()

def plot_data(df, metric='Train/Loss', title='Loss', xlabel='Epoch', ylabel='Loss', legend_label='', hold_on=False):
    global figure_number
    # keep only the the rows where the tag == metric
    df = df[df['tag'] == metric]

    values = df['value'].values
    fig = plt.figure(figure_number)
    # plot the values
    plt.plot(values, label=legend_label)
    
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


        
def make_augmentation_comparison_graph():
    x = np.arange(0.0, 17.0, 1)
    # faster, faster_v2, retina, retina_v2, ssd300, ssdlite
    y = np.array([0.8925, 0.9035, 0, 0.9403, 0.9399, 0, 0.8853, 0.9181, 0, 0.8603, 0.9270, 0, 0.6548, 0.7635, 0, 0.3047, 0.2221])
    # colors = ['blue', 'orange', 'b', 'blue', 'orange', 'b', 'blue', 'orange', 'b', 'blue', 'orange', 'b', 'blue', 'orange', 'b', 'blue', 'orange', ]
    colors = [ 'C0', 'C1', 'C1', 'C0', 'C1', 'C1', 'C0', 'C1', 'C1', 'C0', 'C1', 'C1', 'C0', 'C1', 'C1',]
    fig = plt.figure(figure_number)
    plt.bar(x, y, color=colors)
    x += 0.5
    plt.xlabel('Model')
    plt.ylabel('mAP')
    plt.title('Augmentation Comparison')
    # plt.xticks(x, ('A', ' ', ' ', 'B', ' ', ' ', 'C', ' ', ' ', 'D', ' ', ' ',  'E', ' ', ' ', 'F', ' '))
    plt.xticks(x, ('Faster R-CNN', ' ', ' ', 'Faster R-CNN_v2', ' ', ' ', 'RetinaNet', ' ', ' ', 'RetinaNet_v2', ' ', ' ',  'SSD300', ' ', ' ', 'SSDLite', ' '))



    auto_minor_locator = AutoMinorLocator(2)
    plt.gca().xaxis.set_minor_locator(auto_minor_locator)

    
    
    

    plt.yticks(np.arange(0, 1.1, 0.1))
    # include the values on the bars
    for i, v in enumerate(y):
        if v != 0:
            plt.text(i, v, f'{v:.3f}', color='black', ha='center', va='bottom')

    # make a legend
    legend_elements = [mpatches.Patch(facecolor='C0', edgecolor='black', label='Without Augmentations'),
                          mpatches.Patch(facecolor='C1', edgecolor='black', label='With Augmentations')]

    plt.legend(handles=legend_elements, loc='upper right')

    # models = ['FasterRCNN', 'FasterRCNN_v2', 'RetinaNet', 'RetinaNet_v2', 'SSD300', 'SSDLite']
    # letters = ['A', 'B', 'C', 'D', 'E', 'F']
    # text = ''
    # for i in range(len(models)):
    #     text += f'{letters[i]}: {models[i]}\n'
    # # make annotation in top right corner
    # plt.annotate(text, xy=(0.83, 0.5), xycoords='axes fraction')

def make_time_comparison_graph():
    x = np.arange(0, 6, 1)
    # faster, faster_v2, retina, retina_v2, ssd300, ssdlite
    y = np.array([75.48346, 75.28214, 63.39934, 62.17425, 38.86057, 42.37621])
    y = y / 500
    # y is the time it took for 1 image to be processed. Convert it to images per second
    y = 1 / y
    print(y)


    fig = plt.figure(figure_number)
    # make all bars different colors
    plt.bar(x, y)
    # plt.plot(x, y)
    plt.xlabel('Model')
    plt.ylabel('Images per Second')
    plt.title('Time Comparison')
    # plt.xticks(x, ('faster', 'faster_v2', 'retina', 'retina_v2', 'ssd300', 'ssdlite'))
    plt.xticks(x, ('A', 'B', 'C', 'D', 'E', 'F'))
    # make a legend that shows that A = FasterRCNN, B = FasterRCNN_v2, etc.
    letters = ['A', 'B', 'C', 'D', 'E', 'F']
    models = ['FasterRCNN', 'FasterRCNN_v2', 'RetinaNet', 'RetinaNet_v2', 'SSD300', 'SSDLite']
    
    # annotate letters and models in the top left corner

    # make a text with the letters and models
    text = ''
    for letter, model in zip(letters, models):
        text += f'{letter}: {model}\n'

    plt.annotate(text, xy=(0.01, 0.99), xycoords='axes fraction',
            horizontalalignment='left', verticalalignment='top')

    for i, v in enumerate(y):
        if v != 0:
            plt.text(i, v, f'{v:.4f}', color='black', ha='center', va='bottom')

    # save the figure
    # plt.savefig('time_comparison.png', dpi=300, bbox_inches='tight', dir='C:/Users/emilb/OneDrive/Skrivebord/The Master Bitch!/Images for report/figures')



if __name__ == '__main__':
    # data = retrieve_data(EXPERIMENT_ID_RETINA)
    # # plot_data(df=data, metric='Train/Loss', title='Loss', xlabel='Epoch', ylabel='Loss')
    # # plot_data(df=data, metric='Train/Loss_box_reg', title='Loss_box_reg', xlabel='Epoch', ylabel='Loss Box Reggression')
    # # # plot_data(df=data, metric='Train/Loss_classifier', title='Loss_classifier', xlabel='Epoch', ylabel='Loss Classifier')
    # # # plot_data(df=data, metric='Train/Loss_objectness', title='Loss_objectness', xlabel='Epoch', ylabel='Loss Objectness')
    # # # plot_data(df=data, metric='Train/Loss_rpn_box_reg', title='Loss_rpn_box_reg', xlabel='Epoch', ylabel='Loss RPN Box Reggression')

    # # # Validation/mAP with hold_on=True
    # plot_data(df=data, metric='Validation/mAP', title='mAP', xlabel='Epoch', ylabel='mAP', legend_label='RetinaNet', hold_on=True)
    # # plot_data(df=data, metric='Validation/mAP_50', title='mAP_50', xlabel='Epoch', ylabel='mAP_50', hold_on=True)
    # # plot_data(df=data, metric='Validation/mAP_75', title='mAP_75', xlabel='Epoch', ylabel='mAP_75')

    # data = retrieve_data(EXPERIMENT_ID_RETINA_AUG)
    # # PLOT MAP
    # plot_data(df=data, metric='Validation/mAP', title='mAP', xlabel='Epoch', ylabel='mAP', legend_label='RetinaNet_aug', hold_on=False)


    # make_augmentation_graphs()

    # make_augmentation_comparison_graph() 
    # make_time_comparison_graph()   
    test(EXPERIMENT_ID_HP_TUNING_1, 'SSDlite')
    test(EXPERIMENT_ID_HP_TUNING_2, 'SSD300')
    test(EXPERIMENT_ID_HP_TUNING_3, 'Faster R-CNN')
    test(EXPERIMENT_ID_HP_TUNING_4, 'Faster R-CNN_v2')
    test(EXPERIMENT_ID_HP_TUNING_5, 'RetinaNet')
    test(EXPERIMENT_ID_HP_TUNING_6, 'RetinaNet_v2')

    plt.show()

