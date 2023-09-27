from sklearn.tree import export_graphviz
from graphviz import Source
import matplotlib.pyplot as plt
import tempfile

def save_tree(dtree, feature_names, class_names): 
    temp = tempfile.TemporaryFile(suffix='.dot')
    temp.close()
    export_graphviz(
       decision_tree=dtree,
       out_file=temp.name,
       feature_names=feature_names,
       class_names=class_names,
       rounded=True,
       filled=True
    )
    return temp.name

def visualize_tree(dtree, feature_names, class_names): 
    file_name = save_tree(dtree, feature_names, class_names)
    return Source.from_file(file_name)

def plot_results(train_score, test_score, train_label = None, test_label = None, xlabel = None, ylabel = None, xvalues = None):
    if not xvalues: xvalues = range(1, len(train_score) + 1)
    plt.plot(xvalues,train_score,'-', label = train_label)
    plt.plot(xvalues,test_score,'-', label = test_label)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    if xlabel or ylabel: plt.legend()

def grid_plot(elements, width, height, make_plot, figsize = (20, 20)):
    if (width * height < len(elements)): raise ValueError("Insufficient room in the grid")
    fig, axes = plt.subplots(width, height, figsize = figsize)
    for i, element in enumerate(elements):
        ax=axes[i // width][i % width]
        make_plot(ax, element)
    return axes
