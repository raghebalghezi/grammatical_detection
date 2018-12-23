#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 01:11:27 2018

@author: raghebal-ghezi
"""
import nltk
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.manifold import TSNE

def posify(sent, flag='mixed'):
    #returns a pos-tagged sequences, or pos-word sequence.

    tokenized = nltk.tokenize.word_tokenize(sent)

    pos = nltk.pos_tag(tokenized)

    mixed_list = list()
    for t in pos:
        if t[1] in ['CC','IN','RB']:
            mixed_list.append(t[0])
        else:
            mixed_list.append(t[1])
    if flag == 'words': # returns the sentence intact
        return ' '.join(tokenized)
    if flag == 'POS': # returns POS tagged sequence instead
        return ' '.join([t[1] for t in pos])
    if flag == 'mixed': # a mixed of both
        return ' '.join(mixed_list)
    
def convert(label):
    dic = {"A":1,"B":2,"C":3}
#    dic = {"A1":1,"A2":2,"B1":3,"B2":4,"C1":5,"C2":6}
    return dic[str(label)]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def project(texts_list, num_clusters):
    # project and plot. adapted from http://www.itkeyword.com/doc/7191219466270725017/how-do-i-visualize-data-points-of-tf-idf-vectors-for-kmeans-clustering
    num_seeds = 10
    max_iterations = 300
    labels_color_map = {
        0: '#001eff', 1: '#006865', 2: '#ff001a', 3: '#005073', 4: '#4d0404',
        5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'
    }
    pca_num_components = 2
    tsne_num_components = 2
    
    # texts_list = some array of strings for which TF-IDF is being computed
    
    # calculate tf-idf of texts
    tf_idf_vectorizer = TfidfVectorizer(analyzer="word", use_idf=True, smooth_idf=True, ngram_range=(2, 3))
    tf_idf_matrix = tf_idf_vectorizer.fit_transform(texts_list)
    
    # create k-means model with custom config
    clustering_model = KMeans(
        n_clusters=num_clusters,
        max_iter=max_iterations,
        precompute_distances="auto",
        n_jobs=-1
    )
    
    labels = clustering_model.fit_predict(tf_idf_matrix)
    # print labels
    
    X = tf_idf_matrix.todense()
    
    # ----------------------------------------------------------------------------------------------------------------------
    
    reduced_data = PCA(n_components=pca_num_components).fit_transform(X)
    # print reduced_data
    
    fig, ax = plt.subplots()
    for index, instance in enumerate(reduced_data):
        # print instance, index, labels[index]
        pca_comp_1, pca_comp_2 = reduced_data[index]
        color = labels_color_map[labels[index]]
        ax.scatter(pca_comp_1, pca_comp_2, c=color)
    plt.show()
    
    
#    
#    # t-SNE plot
#    embeddings = TSNE(n_components=tsne_num_components)
#    Y = embeddings.fit_transform(X)
#    plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
#    plt.show()


def plot_pres_recall(classifier,X_train, X_test, y_train, y_test):
    
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    
    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(3):
        precision[i], recall[i], _ = precision_recall_curve(y_test[i],
                                                            y_score[i])
        average_precision[i] = average_precision_score(y_test[i], y_score[i])
    
    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score,
                                                         average="micro")
    
    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall[0], precision[0], label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
    plt.legend(loc="lower left")
    plt.show()
    
    # Plot Precision-Recall curve for each class
    plt.clf()
    plt.plot(recall["micro"], precision["micro"],
             label='micro-average Precision-recall curve (area = {0:0.2f})'
                   ''.format(average_precision["micro"]))
    for i in range(3):
        plt.plot(recall[i], precision[i],
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(i, average_precision[i]))
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(loc="lower right")
    plt.show()