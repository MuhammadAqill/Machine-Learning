import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

class JumpingSmashSVM:
    def __init__(self, file_path):
        """
        Initialize the JumpingSmashSVM with the path to the data file.
        
        Args:
            file_path (str): The path to the Excel file containing the data.
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
        self.imputer = None

    def load_data(self):
        """Load data from the provided Excel file into a pandas DataFrame."""
        self.data = pd.read_excel(self.file_path, sheet_name=None)
        self.combined_data = pd.concat(self.data.values(), ignore_index=True)

    def preprocess_data(self):
        """
        Prepare features and labels for training.
        Extract relevant columns for features and assign labels.
        Handle NaN values by imputing missing data with the mean.
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
        
        self.features = self.combined_data[feature_columns]
        self.labels = self.combined_data['Phase']  # Assuming 'Phase' is the correct label column
        
        # Imputasi nilai NaN dengan rata-rata
        self.imputer = SimpleImputer(strategy='mean')
        self.features = self.imputer.fit_transform(self.features)

    def split_data(self):
        """
        Split the dataset into training and testing sets.
        Uses 80% of the data for training and 20% for testing.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=42, stratify=self.labels
        )

    def scale_data(self):
        """Normalize the feature data using StandardScaler."""
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_model(self):
        """Train the Support Vector Machine (SVM) model using the training data."""
        self.model = SVC(kernel='linear', random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """
        Evaluate the trained model using the test data.
        Print accuracy, classification report, and confusion matrix.
        """
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        print(classification_report(self.y_test, y_pred))

        # Confusion Matrix
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        self.plot_confusion_matrix(conf_matrix)

    def plot_confusion_matrix(self, conf_matrix):
        """
        Plot the confusion matrix using seaborn's heatmap.
        
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
        """Execute the entire process: load data, preprocess, train, and evaluate the model."""
        self.load_data()
        self.preprocess_data()
        self.split_data()
        self.scale_data()
        self.train_model()
        self.evaluate_model()


# Main execution
if __name__ == "__main__":
    # Example file path, adjust as needed
    file_path = 'output_with_phases_all_sheets.xlsx'  # Replace with your actual file path
    jumping_smash_svm = JumpingSmashSVM(file_path)
    jumping_smash_svm.run()
