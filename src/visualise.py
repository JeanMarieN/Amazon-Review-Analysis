import matplotlib.pyplot as plt
import sklearn.metrics
import itertools
import numpy as np

def plot_results(losses, accuracies, title, output_path):
    fig, ax = plt.subplots(1,2,figsize=[12,2])
    ax[0].plot(losses)
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('iteration')
    ax[1].plot(accuracies)
    ax[1].set_ylabel('accuracy')
    ax[1].set_xlabel('iteration')
    fig.suptitle(title, fontsize=16)
    plt.savefig(output_path)

def plot_both_results(losses, accuracies, val_losses, val_accuracies, title, output_path):
    fig, ax = plt.subplots(1,2,figsize=[12,2])
    ax[0].plot(losses, marker='', color='grey', linewidth=2, label='Training')
    ax[0].plot(val_losses, marker='', color='orange', linewidth=2, label='Validation')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Iteration')
    ax[1].plot(accuracies, marker='', color='grey', linewidth=2, label='Training')
    ax[1].plot(val_accuracies, marker='', color='orange', linewidth=2, label='Validation')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Iteration')
    fig.suptitle(title, fontsize=16)
    plt.savefig(output_path)
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    From: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')
#
#    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] != 0:
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    
def plot_confusion(y, y_pred, label_list) -> None:
    """
    Args:
        y: true labels
        y_pred: predicted labels
        label_list: ordered iterable of labels
    """
    # Compute confusion matrix
    cnf_matrix = sklearn.metrics.confusion_matrix(y, y_pred, labels=label_list)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plt.figure(figsize=(13,10))
    plot_confusion_matrix(cnf_matrix, classes=label_list,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plt.figure(figsize=(13,10))
    plot_confusion_matrix(cnf_matrix, classes=label_list, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()