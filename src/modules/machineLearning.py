from re import S
from src.tools.machine_learning.featureSelector import FeatureSelector
from src.tools.machine_learning.dataSplit import DataSplit
from src.tools.machine_learning.algorithms import Algorithms
from src.tools.machine_learning.visualizer import Visualizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class MachineLearning:

    def __init__ (self, json, df):
        self.ml = json['machine_learning']
        self.df = df
        (self.X, self.Y, self.X_train, self.X_test, self.Y_train, self.Y_test) = self._prepare_data()

    def _prepare_data(self):

        X = self.df.loc[:, self.df.columns != self.ml['target_var']]
        Y = self.df.loc[:, self.df.columns == self.ml['target_var']]

        ds = DataSplit(self.ml['df_test_size'])
        X_train, X_test, Y_train, Y_test = ds.train_test_split(X, Y)


        fs = FeatureSelector(self.ml['target_var'], X_train, Y_train)
        fs.correlation_analysis_RandForest(self.ml['ntrees'], log_scale = False)

        return (X, Y, X_train, X_test, Y_train, Y_test)

    def _build_random_forest_class_model(self):

        print('\n****\nRandom Forest Classifier model data\n****')
        
        data = Algorithms(self.X_train, self.X_test, self.Y_train, self.Y_test)

        results_rfc = data.random_forest_class_model(self.ml['ntrees'])

        rfc_model = results_rfc['model']
        Y_pred = results_rfc['Y_pred']
        Visualizer(rfc_model).visualize_classification_results(
            self.X_test, self.Y_test, Y_pred, self.ml['target_var_values']
        )

        return rfc_model

    def _build_SVM_model(self):

        print('\n****\nSVM model data\n****')
        data = Algorithms(self.X_train, self.X_test, self.Y_train, self.Y_test)
        results_svm = data.svm_model()
        
        svm_model = results_svm['model']
        Y_pred = results_svm['Y_pred']
        Visualizer(svm_model).visualize_classification_results(
            self.X_test, self.Y_test, Y_pred, self.ml['target_var_values']
        )

        return svm_model

    def build_ml_models(self):
        data = Algorithms(self.X_train, self.X_test, self.Y_train, self.Y_test)

        print('\n****\nCross val for RFC\n****')
        rfc_tool = RandomForestClassifier(n_estimators = 1000, n_jobs = -1)
        data.cross_val_score_classification(rfc_tool, 5)
        model1 = self._build_random_forest_class_model()

        print('\n****\nCross val for SVM\n****')
        svc_tool = SVC()
        data.cross_val_score_classification(svc_tool, 5)
        model2 = self._build_SVM_model()
        return (model1, model2)