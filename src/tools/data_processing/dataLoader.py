from IPython.display import Image, display
import pandas as pd

class DataLoader():

    def __init__(self, path):
        self.path = path

    def diplay_toc(self):
        img = Image(self.path)
        display(img)

    def load_data(self, blank_replacement = '_'):
        database = pd.read_csv(self.path)
        database = database.rename(columns = lambda x: x.strip())
        database = database.rename(columns = lambda x: x.capitalize().replace(' ', blank_replacement))
        print(f'Dimensionality: {database.shape}')
        print(f'Variables:\n {database.columns}')
        return database
