import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
class IrisKNNApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Set window properties
        self.setWindowTitle('Iris KNN Classifier')
        self.setGeometry(100, 100, 600, 400)

        # Layout and widgets
        layout = QVBoxLayout()

        self.load_button = QPushButton('Load Iris Dataset', self)
        self.load_button.clicked.connect(self.load_data)
        layout.addWidget(self.load_button)

        self.train_button = QPushButton('Train KNN Model', self)
        self.train_button.clicked.connect(self.train_model)
        self.train_button.setEnabled(False)  # Disable until data is loaded
        layout.addWidget(self.train_button)

        self.result_label = QLabel('Results will be displayed here.', self)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def load_data(self):
        # Load dataset using a file dialog
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Iris Dataset CSV", "", "CSV Files (*.csv)", options=options)
        if file_name:
            try:
                self.iris_data = pd.read_csv(file_name)
                self.result_label.setText('Dataset loaded successfully.')
                self.train_button.setEnabled(True)  # Enable training button
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Could not load the dataset: {e}')

    def train_model(self):
        try:
            X = self.iris_data.drop('Species', axis=1)
            y = self.iris_data['Species']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train, y_train)

            y_pred = knn.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            self.result_label.setText(f'Accuracy: {accuracy:.2f}\n\nClassification Report:\n{report}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Error training the model: {e}')
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = IrisKNNApp()
    ex.show()
    sys.exit(app.exec_())
