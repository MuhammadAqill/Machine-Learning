import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class JumpingSmashRandomForestModel:
    def __init__(self, file_path):
        """
        Initialize the model with the path to the dataset.
        :param file_path: str, path to the Excel file containing the dataset
        """
        self.file_path = file_path
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

    def load_data(self):
        """
        Load data from the provided Excel file, combining all sheets into one DataFrame.
        """
        self.data = pd.read_excel(self.file_path, sheet_name=None)
        self.combined_data = pd.concat(self.data.values(), ignore_index=True)

    def preprocess_data(self):
        """
        Prepare the feature set and labels for training.
        Assumes that the relevant features and target are correctly named in the dataset.
        """
        self.features = self.combined_data[['Left Hip-Knee Angle ', 'Right Hip-Knee Angle ', 
                                            'Left Knee-Ankle Angle ', 'Right Knee-Ankle Angle ',
                                            'Left Hip-Knee Ang Vel', 'Right Hip-Knee Ang Vel',
                                            'Left Knee-Ankle Ang Vel', 'Right Knee-Ankle Ang Vel',
                                            'Left Hip-Knee Ang Acc', 'Right Hip-Knee Ang Acc',
                                            'Left Knee-Ankle Ang Acc', 'Right Hip-Knee Ang Acc.1',
                                            'Left Hip Acc L (X)', 'Left Hip Acc L (Y)', 'Left Hip Acc L (Z)',
                                            'Right Hip Acc L (X)', 'Right Hip Acc L (Y)', 'Right Hip Acc L (Z)',
                                            'Left Knee Acc L (X)', 'Left Knee Acc L (Y)', 'Left Knee Acc L (Z)',
                                            'Right Knee Acc L (X)', 'Right Knee Acc L (Y)', 'Right Knee Acc L (Z)',
                                            'Left Ankle Acc L (X)', 'Left Ankle Acc L (Y)', 'Left Ankle Acc L (Z)',
                                            'Right Ankle Acc L (X)', 'Right Ankle Acc L (Y)', 'Right Ankle Acc L (Z)']]
        
        self.labels = self.combined_data['Phase']

    def split_data(self):
        """
        Split the data into training and testing sets.
        The training set will consist of 80% of the data and the test set will consist of 20%.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, 
                                                                                self.labels, 
                                                                                test_size=0.2, 
                                                                                random_state=42, 
                                                                                stratify=self.labels)

    def scale_data(self):
        """
        Normalize the feature data using StandardScaler.
        This ensures that features are on the same scale, which is important for many algorithms.
        """
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_model(self):
        """
        Train the Random Forest model on the training data.
        """
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """
        Evaluate the trained Random Forest model on the test data.
        Prints the accuracy, classification report, and confusion matrix.
        """
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        print(classification_report(self.y_test, y_pred))

        # Plotting Confusion Matrix
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=self.model.classes_, yticklabels=self.model.classes_)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def run(self):
        """
        Execute the entire pipeline from loading data to training and evaluating the model.
        """
        self.load_data()
        self.preprocess_data()
        self.split_data()
        self.scale_data()
        self.train_model()
        self.evaluate_model()


# Main execution
if __name__ == "__main__":
    file_path = 'output_with_phases_all_sheets.xlsx'  # Update with your actual file path
    jumping_smash_model = JumpingSmashRandomForestModel(file_path)
    jumping_smash_model.run()
