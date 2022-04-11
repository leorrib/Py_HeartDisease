import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

class Algorithms():

    def __init__ (self, X_train, X_test, Y_train, Y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

    def _train_Linear_Regression_model(self):
        model = LinearRegression()
        model.fit(self.X_train, self.Y_train)
        r2 = model.score(self.X_train, self.Y_train)
        print(f'R-squared value (training): {r2}')
        return model

    def _test_Linear_Regression_model(self, model):
        Y_pred = model.predict(self.X_test)

        r2 = model.score(self.X_test, self.Y_test)
        rmse = np.sqrt(mean_squared_error(self.Y_test, Y_pred))
        norm_rmse = rmse / (max(self.Y_train) - min(self.Y_train)) * 100
        print(f'R-squared value (testing): {r2}')
        print(f'Root mean squared error: {rmse}')
        print(f'Normalized root mean squared error: {norm_rmse}%')
        return Y_pred

    def linear_regression_model (self):
        data = Algorithms(self.X_train, self.X_test, self.Y_train, self.Y_test)
        model = data._train_Linear_Regression_model()
        Y_pred = data._test_Linear_Regression_model(model)

        result = { "model": model, "Y_pred": Y_pred }

        return result

    def _train_random_forest_regressor_model(self, tool):
        model = tool
        model.fit(self.X_train, self.Y_train)
        r2 = model.score(self.X_train, self.Y_train)
        print(f'R-squared value (training): {r2}')
        return model

    def _test_random_forest_regressor_model(self, model):
        Y_pred = model.predict(self.X_test)

        r2 = model.score(self.X_test, self.Y_test)
        rmse = np.sqrt(mean_squared_error(self.Y_test, Y_pred))
        norm_rmse = rmse / (max(self.Y_train) - min(self.Y_train)) * 100
        print(f'R-squared value (testing): {r2}')
        print(f'Root mean squared error: {rmse}')
        print(f'Normalized root mean squared error: {norm_rmse}%')
        return Y_pred

    def random_forest_regressor_model (self):
        data = Algorithms(self.X_train, self.X_test, self.Y_train, self.Y_test)
        tool = RandomForestRegressor()
        model = data._train_random_forest_regressor_model(tool)
        Y_pred = data._test_random_forest_regressor_model(model)

        result = { "model": model, "Y_pred": Y_pred }

        return result

    def _train_random_forest_class_model (self, ntrees):
        ml_tool = RandomForestClassifier(n_estimators = ntrees)
        model = ml_tool.fit(self.X_train, self.Y_train)

        return model

    def _test_random_forest_class_model (self, model):
        Y_pred = model.predict(self.X_test)
        acc = accuracy_score(self.Y_test, Y_pred)
        print(f'Accuracy: {acc}')

        return Y_pred

    def random_forest_class_model (self, ntrees):
        data = Algorithms(self.X_train, self.X_test, self.Y_train, self.Y_test)
        model = data._train_random_forest_class_model(ntrees)
        Y_pred = data._test_random_forest_class_model(model)

        result = { "model": model, "Y_pred": Y_pred }

        return result

    def _train_SVM_model (self):
        ml_tool = SVC()
        model = ml_tool.fit(self.X_train, self.Y_train)

        return model

    def _test_SVM_model (self, model):
        Y_pred = model.predict(self.X_test)
        acc = accuracy_score(self.Y_test, Y_pred)
        print(f'Accuracy: {acc}')

        return Y_pred

    def svm_model (self):
        data = Algorithms(self.X_train, self.X_test, self.Y_train, self.Y_test)
        model = data._train_SVM_model()
        Y_pred = data._test_SVM_model(model)

        result = { "model": model, "Y_pred": Y_pred }

        return result

    def cross_val_score_classification(self, model, n_folds):

        kfold = KFold(n_splits = n_folds, shuffle = True)

        # score = cross_val_score(modelo, X, Y, cv = kfold)
        acc = cross_val_score(model, self.X_test, self.Y_test, cv = kfold, scoring = 'accuracy')
        auc = cross_val_score(model, self.X_test, self.Y_test, cv = kfold, scoring = 'roc_auc')

        print('Mean accuracy: %.3f' % np.mean(acc))
        print('Mean AUC: %.3f' % np.mean(auc))

    def cross_val_score_regression(self, model):

        n_folds = 5
        kfold = KFold(n_splits = n_folds, shuffle = True)

        # score = cross_val_score(modelo, X, Y, cv = kfold)
        r2 = cross_val_score(model, self.X_train, self.Y_train, cv = kfold, scoring = 'r2')
        rmse = cross_val_score(
            model, self.X_train, self.Y_train, cv = kfold, scoring = 'neg_root_mean_squared_error'
        )

        norm_rmse = -np.mean(rmse / (max(self.Y_train) - min(self.Y_train))) * 100
        print(f'R-squared value: {np.mean(r2)}')
        print(f'Root mean squared error: {-np.mean(rmse)}')
        print(f'Normalized root mean squared error: {norm_rmse}%')