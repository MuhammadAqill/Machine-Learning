import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer


class KNNModel:
    def __init__(self, file_path, k=5):
        """
        Initializes the KNNModel with the file path for the dataset and the number of neighbors for the KNN classifier.
        
        Args:
        file_path (str): Path to the Excel file containing the dataset.
        k (int): The number of neighbors to use for KNN classification (default is 5).
        """
        self.file_path = file_path
        self.k = k
        self.data = None
        self.combined_data = None
        self.features = None
        self.labels = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = None
        self.imputer = SimpleImputer(strategy='mean')  # Impute missing values with the mean

    def load_data(self):
        """
        Loads the dataset from the provided Excel file path and combines all sheets into one DataFrame.
        """
        try:
            self.data = pd.read_excel(self.file_path, sheet_name=None)
            self.combined_data = pd.concat(self.data.values(), ignore_index=True)
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def preprocess_data(self):
        """
        Preprocess the dataset by selecting feature columns, imputing missing values, 
        and extracting features and labels for training and testing.
        """
        feature_columns = [
            'Left Hip-Knee Angle ', 'Right Hip-Knee Angle ', 'Left Knee-Ankle Angle ', 'Right Knee-Ankle Angle ',
            'Left Hip-Knee Ang Vel', 'Right Hip-Knee Ang Vel', 'Left Knee-Ankle Ang Vel', 'Right Knee-Ankle Ang Vel',
            'Left Hip-Knee Ang Acc', 'Right Hip-Knee Ang Acc', 'Left Knee-Ankle Ang Acc', 'Right Hip-Knee Ang Acc.1',
            'Left Hip Acc L (X)', 'Left Hip Acc L (Y)', 'Left Hip Acc L (Z)', 'Right Hip Acc L (X)', 
            'Right Hip Acc L (Y)', 'Right Hip Acc L (Z)', 'Left Knee Acc L (X)', 'Left Knee Acc L (Y)', 
            'Left Knee Acc L (Z)', 'Right Knee Acc L (X)', 'Right Knee Acc L (Y)', 'Right Knee Acc L (Z)', 
            'Left Ankle Acc L (X)', 'Left Ankle Acc L (Y)', 'Left Ankle Acc L (Z)', 'Right Ankle Acc L (X)', 
            'Right Ankle Acc L (Y)', 'Right Ankle Acc L (Z)'
        ]
        
        # Extract features and labels from the combined dataset
        self.features = self.combined_data[feature_columns]
        self.labels = self.combined_data['Phase']

        # Impute missing values in features
        self.features = self.imputer.fit_transform(self.features)

    def split_data(self):
        """
        Splits the data into training and testing sets.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=42, stratify=self.labels
        )

    def scale_data(self):
        """
        Standardizes the feature data by scaling it to have a mean of 0 and variance of 1.
        """
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_model(self):
        """
        Initializes and trains the KNN model using the training data.
        """
        self.model = KNeighborsClassifier(n_neighbors=self.k)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """
        Evaluates the trained KNN model on the test set and prints the accuracy, classification report, and confusion matrix.
        """
        y_pred = self.model.predict(self.X_test)

        # Evaluate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        print(classification_report(self.y_test, y_pred))

        # Generate and plot confusion matrix
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        self.plot_confusion_matrix(conf_matrix)

    def plot_confusion_matrix(self, conf_matrix):
        """
        Plots the confusion matrix as a heatmap for better visualization.
        
        Args:
        conf_matrix (ndarray): The confusion matrix to be plotted.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.model.classes_, yticklabels=self.model.classes_)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def run(self):
        """
        Runs the entire pipeline of loading data, preprocessing, training, and evaluation.
        """
        self.load_data()
        self.preprocess_data()
        self.split_data()
        self.scale_data()
        self.train_model()
        self.evaluate_model()


# Main execution block
if __name__ == "__main__":
    # Define the path to the dataset
    file_path = 'output_with_phases_all_sheets.xlsx'

    # Initialize the KNN model with the specified file path and number of neighbors
    knn_model = KNNModel(file_path, k=5)

    # Run the model pipeline
    knn_model.run()
