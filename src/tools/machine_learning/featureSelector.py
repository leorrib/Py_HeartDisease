from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class FeatureSelector():

    def __init__ (self, target_var, X_train, Y_train):
        self.target_var = target_var
        self.X_train = X_train
        self.Y_train = Y_train

    def correlation_analysis_KBest (self, log_scale):

        best_var = SelectKBest(score_func = f_regression, k = 'all')
        fit = best_var.fit(self.X_train, self.Y_train)
        fit.transform(self.X_train)

        labels = self.X_train.columns

        data = {'Index': self.X_train.columns, 'Relevance': fit.scores_} 
        df = pd.DataFrame(data)
        if (log_scale):
            df['Relevance'] = np.log(data['Relevance'])
        df.set_index('Index', inplace = True)

        fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (14, 7))
        bplot = sns.barplot(x = labels, y = fit.scores_, ax = ax1, color = 'grey')
        bplot.set_xticklabels(labels, rotation = 90)
        bplot.set_title('Correlation relevance distribution', fontdict={'fontsize':18}, pad=16)
        
        heatmap = sns.heatmap(df.sort_values(by='Relevance', ascending=False), vmin=3, vmax=9, annot=True, cmap='BrBG', ax = ax2)
        heatmap.set_title(f'Features Correlating with {self.target_var} (log scale)', fontdict={'fontsize':18}, pad=16)

        fig.tight_layout()
        plt.show()

    def correlation_analysis_RandForest (self, ntrees, log_scale):

        labels = self.X_train.columns
        alg = RandomForestClassifier(n_estimators = ntrees)
        model = alg.fit(self.X_train, self.Y_train)

        data = {'Index': self.X_train.columns, 'Relevance': model.feature_importances_} 
        df = pd.DataFrame(data)
        df.set_index('Index', inplace = True)
        if (log_scale):
            df['Relevance'] = np.log(data['Relevance'])
            title = f'Features Correlating with {self.target_var} (log scale)'
        else:
            title = f'Features Correlating with {self.target_var}'

        fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (14, 7))
        bplot = sns.barplot(x = labels, y = model.feature_importances_, ax = ax1, color = 'grey')
        bplot.set_xticklabels(labels, rotation = 90)
        bplot.set_title('Correlation relevance distribution', fontdict={'fontsize':18}, pad=16)
        
        heatmap = sns.heatmap(df.sort_values(by = 'Relevance', ascending = False), 
                              vmin = df.Relevance.min(), vmax = df.Relevance.max(), annot = True, 
                              cmap = 'BrBG', ax = ax2)
        ax2.set_ylabel('')
        ax2.set_xlabel('\nImportance')
        ax2.set_xticks([])
        heatmap.set_title(title, fontdict = {'fontsize': 18}, pad = 16)

        fig.tight_layout()
        plt.show()
