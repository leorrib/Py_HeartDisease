from sklearn.model_selection import train_test_split

class DataSplit():

    def __init__ (self, test_sample_size):
        self.test_sample_size = test_sample_size

    def train_test_split(self, variables, target):
        
        X_train, X_test, Y_train, Y_test = train_test_split(
            variables, target, test_size = self.test_sample_size)

        return (X_train, X_test, Y_train, Y_test)