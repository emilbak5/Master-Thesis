from matplotlib import pyplot as plt
import pandas as pd
import tensorboard as tb


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
    plt.figure(figure_number)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # plot the values
    plt.plot(values)

    if not hold_on:
        figure_number += 1
        




if __name__ == '__main__':
    data = retrieve_data(EXPERIMENT_ID)
    plot_data(df=data, metric='Train/Loss', title='Loss', xlabel='Epoch', ylabel='Loss')
    plot_data(df=data, metric='Train/Loss_box_reg', title='Loss_box_reg', xlabel='Epoch', ylabel='Loss box reggression')

    # Validation/mAP with hold_on=True
    plot_data(df=data, metric='Validation/mAP', title='mAP', xlabel='Epoch', ylabel='mAP', hold_on=True)
    plot_data(df=data, metric='Validation/mAP_50', title='mAP_50', xlabel='Epoch', ylabel='mAP_50', hold_on=True)
    plot_data(df=data, metric='Validation/mAP_75', title='mAP_75', xlabel='Epoch', ylabel='mAP_75')

    plt.show()

