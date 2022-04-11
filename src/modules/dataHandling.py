from src.tools.data_processing.dataHandler import DataHandler
from src.tools.data_processing.dataExplorer import DataExplorer

class DataHandling():

    def __init__ (self, json, df):
        self.dh = json['data_handling']
        self.df = df


    def _manipulate_data (self):

        db = DataHandler(self.df).get_dummy_variables(self.dh['cols_to_split'])

        scaled_data = DataHandler(db).min_max_cols(db.columns)

        return scaled_data

    def handle_data(self):

        df = self._manipulate_data()
        DataExplorer(df).get_strong_corr_predict_vars(self.dh['target_var'], self.dh['corr_cutoff'])
        return df

