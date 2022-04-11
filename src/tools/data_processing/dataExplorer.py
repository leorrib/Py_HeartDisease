import pandas as pd
import numpy as np
import seaborn as sns
import math
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class DataExplorer():

    def __init__(self, df):
        self.df = df

    def find_values(self, value):
        ''''
            Finds the desired value in columns
        '''

        for i in range(len(self.df.columns)):
            colname = self.df.columns[i]
            if value == None:
                S = self.df.iloc[:, i].isna().sum()
            else:
                S = (self.df.iloc[:, i] == value).sum()
            msg = f'The column {colname} has {S} values equal to {value}.'
            if S != 0:
                print(msg) 

    def visualize_target_var(self, graph_specs):
        gs = graph_specs
        histplt = sns.displot(self.df, 
                      x = gs['target_var'], 
                      bins = gs['bins'], 
                      legend = False, 
                      hue = gs['hue'], 
                      palette = gs['colors'])
        edges = [rect.get_x() for rect in histplt.ax.patches] + [histplt.ax.patches[-1].get_x() + histplt.ax.patches[-1].get_width()]
        # mids = [rect.get_x() + rect.get_width() / 2 for rect in histplt.ax.patches]
        histplt.ax.set_xticks(edges)
        histplt.set(xlabel = gs['x_label'], ylabel = gs['y_label'])
        plt.legend(loc = 'best', labels = gs['labels'])
        plt.show()

    def get_strong_corr(self, var_target, cutoff):
        df = self.df.corr(method = 'spearman')
        for j in range(len(df.columns)):
            for i in range(j, len(df)):
                if (abs(df.iloc[i, j]) > cutoff) and (i != j) and (df.columns[j] == var_target or df.index[i] == var_target):
                    print(f'Corr coef between {df.columns[j]} and {df.index[i]}: {df.iloc[i, j]}')

    def plot_corr_heatmap(self):
        cors = self.df.corr(method = 'spearman')
        plt.figure(figsize=(16, 8))
        upper = np.triu(np.ones_like(cors))
        heatmap = sns.heatmap(
            cors, vmin = -1, vmax = 1, annot = True, mask = upper, cmap = 'coolwarm'
        )
        heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12)
        plt.show()

    def multiple_var_plot(self, lista_vars, target_var):

        row_num = math.ceil(len(lista_vars) / 2)
        
        cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
        fig, ax = plt.subplots(row_num, 2, figsize = (16, 8))
            
        num = 0
        if row_num == 1:
            while num < len(lista_vars):
                for j in range(0, 2):
                    sns.scatterplot(
                        data = self.df, x = lista_vars[num], y = target_var, 
                        hue = target_var, palette = cmap, ax = ax[j]
                    )
                    num = num + 1
                    if num == len(lista_vars): break
        else:
            while num < len(lista_vars):
                for i in range(0, row_num):
                    for j in range(0, 2):
                        sns.scatterplot(data = self.df, x = lista_vars[num], y = target_var, 
                                        hue = target_var, palette = cmap, ax = ax[i,j])
                        num = num + 1
                        if num == len(lista_vars): break
                    if num == len(lista_vars): break

        plt.show()


    def double_histogram(self, json):
        fig, ax = plt.subplots(ncols = 2, figsize = (12, 6), sharey = True)
    
        df_list = []
        for i in range(len(json['target_var_values'])):
            df_list.append(self.df.loc[self.df[json['target_var']] == json['target_var_values'][i]])
            
        palette = {}
        for i in range(len(json['hue_values'])):
            palette[json['hue_values'][i]] = json['colors'][i]
        
        for i in range(len(df_list)):
            sns.histplot(df_list[i], 
                         x = json['target_var'],
                         hue = json['hue'],
                         hue_order = json['hue_values'],
                         palette = palette,
                         multiple = 'dodge',
                         ax = ax[i])
            ax[i].axes.get_xaxis().set_visible(False)
            ax[i].set_title(json['titles'][i])
        
        plt.show()

    def get_correlation_table (self, cols, file = 'ipynb'):
        dataf = pd.DataFrame(self.df[cols].corr())
        if (file == 'py'):
            print(dataf)
        else:
            display(Markdown(dataf.to_markdown()))

    def get_strong_corr_predict_vars(self, target_var, cutoff):
        df = self.df.copy()
        corr_mat = df.corr(method = 'spearman')
        for j in range(len(corr_mat.columns)):
            for i in range(j, len(corr_mat)): 
                if (abs(corr_mat.iloc[i, j] > cutoff) and (i != j) and 
                    corr_mat.columns[j] != target_var and corr_mat.index[i] != target_var):
                    print(f"Corr coef between {corr_mat.columns[j]} and {corr_mat.index[i]}: {corr_mat.iloc[i, j]}")