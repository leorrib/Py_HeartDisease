from src.modules.dataAnalyzing import DataAnalyzing
from src.modules.dataHandling import DataHandling
from src.modules.machineLearning import MachineLearning
import json
with open('./parameters.json') as json_file:
    parameters = json.load(json_file)

da = DataAnalyzing(parameters)
df = da.analyze_data('py')

dh = DataHandling(parameters, df)
df2 = dh.handle_data()

ml = MachineLearning(parameters, df2)
ml.build_ml_models()