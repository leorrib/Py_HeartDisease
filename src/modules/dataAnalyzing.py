from src.tools.data_processing.dataHandler import DataHandler
from src.tools.data_processing.dataLoader import DataLoader
from src.tools.data_processing.dataExplorer import DataExplorer

class DataAnalyzing:

    def __init__ (self, json):
        self.dp = json['data_processing']

    def _loading_data(self):
        DataLoader(self.dp['toc_path']).diplay_toc()
        database = DataLoader(self.dp['db_path']).load_data()
        return database

    def _exploring_data(self, df, file_type):

        de = DataExplorer(df)
        de.find_values(None)
        de.find_values(0)
        database = DataHandler(df).drop_values_from_cols(0, self.dp['cols_to_drop_zero'])

        de2 = DataExplorer(database)
        for i in self.dp['factor_vars_corr']:
            de2.double_histogram(json = self.dp['data_vis_hist'][i])
                
        de2.get_correlation_table(self.dp["numeric_vars_corr"], file_type)

        return database

    def analyze_data(self, file):
            df = self._loading_data()
            dataf = self._exploring_data(df, file)
            return dataf