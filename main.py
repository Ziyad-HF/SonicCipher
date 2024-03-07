import threading
import librosa
from PyQt5.uic import loadUiType
from os import path
from PyQt5.QtWidgets import QMainWindow, QSpacerItem
import PyQt5.QtWidgets as qtw
from PyQt5 import QtCore as qtc
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from scipy.signal import spectrogram
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QPixmap
import joblib
from helpers import extract_passwords_features, extract_person_features, record_audio
from prediction import SvcModel, GbcModel
FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), "main.ui"))


def create_figure(layout):
    figure = Figure()
    canvas = FigureCanvas(figure)
    layout.addWidget(canvas)
    return figure, canvas


def spectro_gram(data, sample_rate, figure, canvas, name):
    # Compute the spectrogram using np.fft.fft
    frequencies, times, power_spectral_density = spectrogram(data, fs=sample_rate, nperseg=1000, noverlap=500)

    # Clear previous plot and plot the new spectrogram
    figure.clear()
    subplot_to_draw_in = figure.add_subplot(111)

    # Add a small constant to avoid zero values before taking the logarithm
    epsilon = 1e-30

    log_spectrogram = 10 * np.log10(np.abs(power_spectral_density) + epsilon)

    # Use pcolormesh to plot the spectrogram
    subplot_to_draw_in.pcolormesh(times, frequencies, log_spectrogram, shading='auto', cmap='viridis')

    subplot_to_draw_in.set_ylabel('Frequency (Hz)')
    subplot_to_draw_in.set_xlabel('Time (s)')
    subplot_to_draw_in.set_title(f'{name} Spectrogram by dB', fontsize=20)

    # Add color bar to the plot
    color_bar = figure.colorbar(
        subplot_to_draw_in.pcolormesh(times, frequencies, log_spectrogram, shading='auto', cmap='viridis'),
        ax=subplot_to_draw_in, format='%+2.0f dB')
    color_bar.set_label('Intensity (dB)')

    # Draw the plot
    canvas.draw()


class MainApp(QMainWindow, FORM_CLASS):

    def __init__(self, parent=None):

        super(MainApp, self).__init__(parent)
        QMainWindow.__init__(self, parent=None)
        self.setupUi(self)
        self.setWindowTitle("Fingerprint Recognition")

        self.passwords_model = None
        self.persons_model = None
        self.create_model()

        self.stats_labels = []
        self.person_probability = None
        self.password_probability = None

        self.peoples_list = self.persons_model.my_classes()
        self.accessed_peoples = []

        self.passwords_list = self.passwords_model.my_classes()
        self.accessed_passwords = []

        self.comboBoxMapper = qtc.QSignalMapper(self)
        self.create_combo_boxes(self.personCheckLayout, self.peoples_list)
        self.create_combo_boxes(self.passwordsLayout, self.passwords_list)

        self.figure, self.canvas = create_figure(self.specrtoLayout)

        self.recordingBtn.clicked.connect(self.recognize_audio)

        self.iconLabel.setPixmap(QPixmap("assets/locked.png").scaled(170, 170))

        self.predicted_person = None
        self.predicted_password = None

    def create_model(self):
        self.passwords_model = joblib.load("password_model.joblib")
        print(f"passwords model accuracy: {self.passwords_model.evaluate()}")

        self.persons_model = joblib.load("person_model.joblib")
        print(f"persons model accuracy: {self.persons_model.evaluate()}")

    def recognize_audio(self):
        saving_path = record_audio()

        data, sr = librosa.load(saving_path)
        spectro_gram(data, sr, self.figure, self.canvas, "Record")

        prediction_thread = threading.Thread(target=self.get_prediction,
                                             args=(saving_path, data, sr))
        prediction_thread.start()
        prediction_thread.join()

        if self.predicted_person in self.accessed_peoples and self.predicted_password in self.accessed_passwords:
            self.messageLabel.setText("Access gained, Hello " + self.predicted_person)

            self.iconLabel.setPixmap(QPixmap("assets/unlocked.png").scaled(170, 170))
        else:
            self.messageLabel.setText("Access denied, try again")
            self.iconLabel.setPixmap(QPixmap("assets/locked.png").scaled(170, 170))

    def get_prediction(self, predicted_path, data, sr):

        self.person_probability = self.persons_model.model.predict_proba([extract_person_features(path=predicted_path,
                                                                                                  audio_file=data,
                                                                                                  sr=sr)])[0]
        max_person_prob = max(self.person_probability)
        if max_person_prob < 0.6:
            self.predicted_person = "Others"
        else:
            self.predicted_person = self.persons_model.my_classes()[np.argmax(self.person_probability)]

        for label, prob in zip(self.stats_labels[:len(self.persons_model.my_classes())], self.person_probability):
            label.setText('{:.2f}'.format(prob * 100))

        print(self.predicted_person)

        self.password_probability = \
            self.passwords_model.model.predict_proba([extract_passwords_features(path=predicted_path,
                                                                                 audio_file=data,
                                                                                 sr=sr)])[0]

        max_password_prob = max(self.password_probability)
        if max_password_prob < 0.8:
            self.predicted_password = "Others"
        else:
            self.predicted_password = self.passwords_model.my_classes()[np.argmax(self.password_probability)]

        for label, prob in zip(self.stats_labels[len(self.persons_model.my_classes()):], self.password_probability):
            label.setText('{:.2f}'.format(prob * 100))

        print(self.predicted_password)

    def create_combo_boxes(self, layout, names_list):
        font = QFont()
        font.setPointSize(12)

        for name in names_list:
            # create a layout for each checkbox and stat label
            stat_layout = qtw.QHBoxLayout()
            stat_layout.setContentsMargins(0, 0, 0, 0)

            # create a label for each stat
            stat_label = qtw.QLabel("0.0")
            stat_label.setFont(font)
            stat_label.setStyleSheet("color: #0e95c2;")
            self.stats_labels.append(stat_label)

            # create a checkbox for each name
            checkbox = qtw.QCheckBox(name)
            checkbox.setFont(font)
            checkbox.setStyleSheet("color: #0e95c2;")

            # create a spacer to push the checkbox and stat label to the right
            spacer = QSpacerItem(40, 20, qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Minimum)
            layout.addItem(spacer)

            # add the checkbox and stat label to the layout
            stat_layout.addWidget(checkbox)
            stat_layout.addItem(spacer)
            stat_layout.addWidget(stat_label)
            layout.addLayout(stat_layout)
            self.comboBoxMapper.setMapping(checkbox, name)
            checkbox.stateChanged.connect(self.checkbox_state_changed)

    def checkbox_state_changed(self, state):
        checkbox = self.sender()
        if state == qtc.Qt.Checked:
            if checkbox.text() in self.peoples_list:
                self.accessed_peoples.append(checkbox.text())
            else:
                self.accessed_passwords.append(checkbox.text())
        else:
            if checkbox.text() in self.peoples_list:
                self.accessed_peoples.remove(checkbox.text())
            else:
                self.accessed_passwords.remove(checkbox.text())


if __name__ == "__main__":
    app = qtw.QApplication([])
    window = MainApp()
    window.show()
    app.exec_()
