from yellowbrick.regressor import ResidualsPlot
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class Visualizer():

    def __init__ (self, model):
        self.model = model

    def visualize_residue_spread(self, X_train, X_test, Y_train, Y_test):
        visualizer = ResidualsPlot(self.model)
        visualizer.fit(X_train, Y_train)
        visualizer.score(X_test, Y_test)
        visualizer.show()

    def _abline(self, slope, intercept):
        plt.style.use('ggplot')
        """Plot a line from slope and intercept"""
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '--', color = 'red')

    def visualize_residue_line(self, Y_test, Y_pred):
        plt.scatter(Y_test, Y_pred, color = 'black')
        Visualizer(self.model)._abline(slope = 1, intercept = 0)
        plt.xlabel('Value computed for target variable')
        plt.ylabel('Real value of the target variable')
        plt.show()

    def visualize_classification_results(self, X_test, Y_test, Y_pred, target_var_names):
        fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (8, 4))

        RocCurveDisplay.from_estimator(self.model, X_test, Y_test, ax = ax1)

        conf_matrix = confusion_matrix(Y_test, Y_pred)

        df = pd.DataFrame(conf_matrix, index = target_var_names, columns = target_var_names)

        hmap = sns.heatmap(df, annot = True, annot_kws = {"fontsize":20}, 
                    vmin = conf_matrix.min(), vmax = conf_matrix.max(), 
                    cmap = 'Blues', ax = ax2, fmt='g')
        hmap.set(ylabel = 'Real', xlabel = 'Predicted')
        fig.tight_layout()
        plt.show()