import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import offsetbox

def plot_embedding_annotation(emb, digits, threshold, ax=None, rescale=True):
    """
    This function is used to visualize the learned low-d features
    We intend to see if we learn to disentangle factors of variations
    @emb : the input low-d feature
    @digits : the immage annotation of emb
    @threshold: minimal distances between two points
    """

    # Rescaling
    if rescale:
        x_min, x_max = np.min(emb, 0), np.max(emb, 0)
        emb = (emb - x_min) / (x_max - x_min)

    _, ax = plt.subplots()

    if hasattr(offsetbox, 'AnnotationBbox'):
        mycanvas = np.array([[1., 1.]])
        for i in range(digits.shape[0]):
            dist = np.sum((emb[i] - mycanvas) ** 2, 1)
            if np.min(dist) < threshold:
                # don't show points that are too close
                # You may try different threshold
                continue
            mycanvas = np.r_[mycanvas, [emb[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits[i], cmap=plt.cm.gray_r),
                emb[i],
                frameon=False)
            ax.add_artist(imagebox)
    ax.set_xticks([])
    ax.set_yticks([])
    return 0


def plot_embedding(emb, labels, ax=None, rescale=False):
    """
    This function is used to visualize the learned low-d features
    We intend to see cluster information via visualization
    @emb : the input low-d feature
    @label : the text annotation of emb
    """
    # Rescaling
    if rescale:
        x_min, x_max = np.min(emb, 0), np.max(emb, 0)
        emb = (emb - x_min) / (x_max - x_min)
    _, ax = plt.subplots()
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']
    for i, _ in enumerate(colors):
        ax.scatter(emb[labels == i, 0], emb[labels == i, 1], c=colors[i], label=i, edgecolors='k')
    ax.legend(scatterpoints=1, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    return 0


def plot_images(X, nrow, ncolumn, rowscale, columnscale):
    """
    This function is used to plot images of the digits
    @X : the input images
    @nrow : number of images per row in canvas
    @ncolumn: number of images per column in canvas
    @rowscale,@columnscale: image scale
    """

    _, ax = plt.subplots()
    imgcanvas = np.zeros(((rowscale+2) * nrow, (columnscale+2) * ncolumn))
    for i in range(nrow):
        ix = (rowscale+2) * i + 1
        for j in range(ncolumn):
            iy = (columnscale+2) * j + 1
            imgcanvas[ix:ix + rowscale, iy:iy + columnscale] = X[i * ncolumn + j]

    ax.imshow(imgcanvas, cmap=plt.cm.binary)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.
    @cm: confusion matrix
    @classes: class names
    @normalize: if True normalize each row
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '%.2f' if normalize else '%d'
    thresh = cm.max() / 2.

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
