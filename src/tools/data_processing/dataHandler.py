from sklearn.preprocessing import MinMaxScaler
from sklearn_pandas import DataFrameMapper
import pandas as pd

class DataHandler():

    def min_max_cols(dataf, cols):
        df = dataf[cols]
        mapper = DataFrameMapper([(df.columns, MinMaxScaler(feature_range = (0, 1)))])
        s_data = mapper.fit_transform(df.copy(), 4)
        scaled_data = pd.DataFrame(s_data, index = df.index, columns = df.columns)
        non_norm_cols = list(set(dataf.columns) - set(df.columns))
        for i in range(len(non_norm_cols)):
            scaled_data[non_norm_cols[i]] = dataf[non_norm_cols[i]]
from sklearn.preprocessing import MinMaxScaler
from sklearn_pandas import DataFrameMapper
import pandas as pd
import numpy as np

class DataHandler():

    def __init__(self, df):
        self.df = df

    def min_max_cols(self, cols):
        df = self.df[cols].copy()
        mapper = DataFrameMapper([(df.columns, MinMaxScaler(feature_range = (0, 1)))])
        s_data = mapper.fit_transform(df.copy(), 4)
        scaled_data = pd.DataFrame(s_data, index = df.index, columns = df.columns)
        non_norm_cols = list(set(self.df.columns) - set(df.columns))
        print(f'Interval of values in the dataframe: [{round(scaled_data.min().min(), 2)}, {round(scaled_data.max().max(), 2)}]')
        for i in range(len(non_norm_cols)):
            scaled_data[non_norm_cols[i]] = self.df[non_norm_cols[i]]
        return scaled_data
                    

    def divide_var_per_range(self, col, numero_de_faixas):

        n = (max(col) - min(col)) / numero_de_faixas
        my_range = np.arange(min(col), max(col) + 2, n + 1)
        df_col = pd.cut(x = col, bins = my_range)
        return df_col.astype(str)

    def get_dummy_variables(self, cols):

        db = self.df.copy()
        cols = ['Sex', 'Chestpaintype', 'Exerciseangina', 'Restingecg', 'St_slope']
        db = pd.get_dummies(db, columns = cols)
        print(db.columns)
        return db

    def drop_values_from_cols(self, value, cols):
        ''''
            Drops rows with certain values in the provided columns
        '''
        df = self.df
        for i in range(len(cols)):
            df = df.drop(df[df[cols[i]] == value].index)
        print(df.shape)
        return df

    def drop_nas(self, cols_to_drop = []):
        ''''
            Drops entire columns, aside from rows with NAs
        '''

        df = self.df.drop(cols_to_drop, axis = 1)
        df = df.dropna(axis = 0)
        print(df.shape)
        return df

    def change_col_names(self, cols_to_rename):
        df = self.df.copy()
        df.rename(columns = cols_to_rename, inplace = True)
        print(f'Variables:\n {df.columns}')
        return df

    def factorize_vars(self, variable_list):
        df = self.df
        for i in range(len(variable_list)):
            df[variable_list[i]] = pd.factorize(df[variable_list[i]])[0]
        return df

    def get_strong_corr_predict_vars(df, target_var, cutoff):
        corr_mat = df.corr(method = 'spearman')
        for j in range(len(corr_mat.columns)):
            for i in range(j, len(corr_mat)): 
                if (abs(corr_mat.iloc[i, j] > cutoff) and (i != j) and 
                    corr_mat.columns[j] != target_var and corr_mat.index[i] != target_var):
                    print(f"Corr coef between {corr_mat.columns[j]} and {corr_mat.index[i]}: {corr_mat.iloc[i, j]}")
                    
