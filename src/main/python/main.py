from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import re
from .background import background
from .Emotion_Detection2 import Emotion_Detection
import numpy as np
import pandas as pd
import pickle
import os
import platform
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, \
    QSlider, QStyle, QSizePolicy, QFileDialog
import sys
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
import subprocess

def read_stylesheet(path_to_sheet):
    with open(path_to_sheet, 'r') as f:
        stylesheet = f.read()
    return stylesheet

class Ui_MainWindow(QtCore.QObject):
    
    def setupUi(self, MainWindow, AppContext):

        # 'Select Video' button
        self.stylesheet_select_unselected = read_stylesheet(AppContext.get_resource('btn_select_unselected.qss'))
        self.stylesheet_select_selected = read_stylesheet(AppContext.get_resource('btn_select_selected.qss'))

        # 'Process' button
        self.stylesheet_process_inactive = read_stylesheet(AppContext.get_resource('btn_process_inactive.qss'))
        self.stylesheet_process_active = read_stylesheet(AppContext.get_resource('btn_process_active.qss'))

        # Progressbar
        self.stylesheet_progressbar_busy = read_stylesheet(AppContext.get_resource('progressbar_busy.qss'))
        self.stylesheet_progressbar_finshed = read_stylesheet(AppContext.get_resource('progressbar_finished.qss'))

        ## Fonts

        # Process inactive
        self.font_asleep = QtGui.QFont('Metropolis', 18)

        # Process active
        self.font_awake = QtGui.QFont('Metropolis', 18)
        self.font_awake.setBold(True)
        

        # ML Models

        self.emotion_model_path = AppContext.get_resource('model.h5') # Path for Emotion Classification Model
        self.prototext = AppContext.get_resource('deploy.prototxt.txt') # Prototxt file for face detection
        self.model = AppContext.get_resource('res10_300x300_ssd_iter_140000.caffemodel') # Model for face Detection
        self.model_path = AppContext.get_resource('finalized_model.sav')
        self.loaded_model = pickle.load(open(self.model_path, 'rb'))

        # Select Video
        font_select = QtGui.QFont('Metropolis', 18)
        font_select.setBold(True)


        ### UI Elements

        dsc_logo_img = AppContext.get_resource('dsc_logo1.png')
        path_logo_small = AppContext.get_resource('GPLogo2.png')
        path_logo = AppContext.get_resource('GenrePrediction2.png')

        self.MainWindow = MainWindow
        self.MainWindow.setObjectName("Genre Prediction")
        self.MainWindow.setStyleSheet("QMainWindow {background:'white'}")
        self.MainWindow.setFixedSize(800, 600)
        self.MainWindow.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)
        self.centralwidget = QtWidgets.QWidget(self.MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.MainWindow.setCentralWidget(self.centralwidget)
        self.MainWindow.setWindowIcon(QtGui.QIcon(path_logo_small))

        # DSC Logo

        dsc_logo = QtWidgets.QLabel(self.centralwidget)
        dsc_logo.setPixmap(QtGui.QPixmap(dsc_logo_img).scaled(540, 100, QtCore.Qt.KeepAspectRatio, transformMode = QtCore.Qt.SmoothTransformation))
        dsc_logo.setObjectName("dsc_logo")
        dsc_logo.setGeometry(300, 10, 1000, 150)

        # Application Mini Logo

        app_mini_logo = QtWidgets.QLabel(self.centralwidget)
        app_mini_logo.setPixmap(QtGui.QPixmap(path_logo_small).scaled(540, 100, QtCore.Qt.KeepAspectRatio, transformMode = QtCore.Qt.SmoothTransformation))
        app_mini_logo.setObjectName("app_mini_logo")
        app_mini_logo.setGeometry(330, -30, 300, 500)

        # Application Name

        app_logo = QtWidgets.QLabel(self.centralwidget)
        app_logo.setPixmap(QtGui.QPixmap(path_logo).scaled(540, 100, QtCore.Qt.KeepAspectRatio, transformMode = QtCore.Qt.SmoothTransformation))
        app_logo.setObjectName("app_logo")
        app_logo.setGeometry(170, 285, 700, 150)

        # Select Video Button

        self.btn_select_video = QtWidgets.QPushButton('Select Video', self.centralwidget)
        self.btn_select_video.setStyleSheet(self.stylesheet_select_unselected)
        self.btn_select_video.setEnabled(True)
        self.btn_select_video.setFixedSize(200, 50)
        self.btn_select_video.setFont(font_select)
        self.btn_select_video.setShortcut('Ctrl+O')
        self.btn_select_video.setGeometry(175, 445, 150, 50)

        # Process Button
        
        self.btn_process = QtWidgets.QPushButton('Process', self.centralwidget)
        self.btn_process.setEnabled(False)
        self.btn_process.setFixedSize(200, 50)
        self.btn_process.setFont(self.font_asleep)
        self.btn_process.setStyleSheet(self.stylesheet_process_inactive)
        self.btn_process.setShortcut('Ctrl+E')
        self.btn_process.setGeometry(435, 445, 150, 50)

        # self.btn_process.clicked.connect(self.Processing)

        # Progress Bar

        self.progress = QtWidgets.QProgressBar(self.MainWindow)
        self.progress.setStyleSheet(self.stylesheet_progressbar_busy)
        self.progress.setGeometry(0, 590, 800, 10)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self.MainWindow)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.MainWindow.setWindowTitle(_translate("MainWindow", "Genre Prediction"))
        self.btn_select_video.clicked.connect(self.open_document)
        self.btn_process.clicked.connect(self.Processing)

        self.MainWindow.show()


    def open_document(self):

        self.video_path = QFileDialog.getOpenFileName(self.MainWindow, 'Open Document', filter = '*.mp4 *.mov *.avi')
        self.video_path = self.video_path[0]
        # print(self.video_path)
        self.output_path = re.sub('mp4', 'avi', self.video_path)
        if self.video_path == '':
            self.sleep_btn_process()
            self.unselect_btn_select()
            return

        self.selected_btn_select()
        self.wake_process()

    
    def Processing(self):

        self.progress.setRange(0, 100)
        self.progress.setStyleSheet(self.stylesheet_progressbar_busy)

        self.input_video = self.video_path # Path for video
        self.c = 0.7 # Confidence score for detecting the face of a person

        background_labels, background_probabilities = background(self.video_path)
        emotion_labels, emotion_probabilities = Emotion_Detection(self.emotion_model_path, self.prototext, self.model, self.video_path, self.c)

        rows = []

        if len(emotion_probabilities) == len(background_probabilities):
            for i in range(0, len(emotion_probabilities)):
                rows.append(emotion_probabilities[i] + background_probabilities[i])  # Concatenating the two lists.

        if rows != []:
            df = pd.DataFrame(rows)
            predictions = list(self.loaded_model.predict(df.values))

            Genres = {0 : 'Horror', 1 : 'Action' , 2 : 'Comedy', 3 : 'Romantic'}

            predictions = list(map(Genres.get, predictions))
            # print(predictions)
            self.popup_success()

            self.final_predictions = predictions

        else:
            self.popup_error()

        cap = cv2.VideoCapture(self.input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)

        seconds_interval = fps * 10
        limit = 0 # A variable used to wait until seconds_interval is reached

        n = len(predictions)
        k = 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total1 = 0
        
        frame_width = int(cap.get(3)) 
        frame_height = int(cap.get(4)) 
        
        size = (frame_width, frame_height)
        result = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
        # print("Output video path is", self.output_path)
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            limit += 1
            total1 += 1
            if(limit != int(seconds_interval) and k < n):
                font = cv2.FONT_HERSHEY_SIMPLEX 
                org = (100, 100) 
                fontScale = 3
                color = (255, 0, 0) 
                thickness = 2
                # Using cv2.putText() method 
                        
                frame = cv2.putText(frame, predictions[k], org, font, fontScale, color, thickness, cv2.LINE_AA)
                # print("Written")
            if limit == int(seconds_interval):
                k += 1
                limit = 0
            result.write(frame)
        cap.release() 
        result.release() 

    def popup_error(self):
        self.stop_progressbar()
        error_popup = QtWidgets.QMessageBox(self.centralwidget)
        error_popup.setIcon(QtWidgets.QMessageBox.Critical)
        error_popup.setWindowTitle('Error: Unable to process video')
        error_popup.setText('Unabel to process video. Raise an issue on the official GDGVIT repo')
        error_popup.setStandardButtons(QtWidgets.QMessageBox.Ok)
        error_popup.show()

    def stop_progressbar(self):
        self.sleep_btn_process()

        self.progress.setRange(0, 1)
        self.progress.setStyleSheet(self.stylesheet_progressbar_finshed)
        self.progress.setTextVisible(False)
        self.progress.setValue(1)
        self.unselect_btn_select()

    def popup_success(self):
        self.stop_progressbar()
        success_popup = QtWidgets.QMessageBox(self.centralwidget)
        success_popup.setIcon(QtWidgets.QMessageBox.NoIcon)
        success_popup.setWindowTitle('Success: File Written')
        success_popup.setText('The Processed Video was successfully saved at ' + self.output_path)
        btn_open_folder = QtWidgets.QPushButton('Play the Processed Video')
        btn_open_folder.clicked.connect(self.showvideo)
        success_popup.addButton(btn_open_folder, QtWidgets.QMessageBox.AcceptRole)
        success_popup.setStandardButtons(QtWidgets.QMessageBox.Ok)
        success_popup.show()

    def open_containing_folder(self):
        if platform.system() == 'Windows':
            video_path = re.search('^(.+)\\([^\\]+)$', self.output_path).groups[0]
            os.startfile(output_path)
            # print(output_path)
        elif platform.system() == 'Darwin':
            output_path = re.search('^(.+)/([^/]+)$', self.output_path).groups()[0]
            # print(output_path)
            subprocess.Popen(['open', output_path])
            
        else:
            output_path = re.search('^(.+)/([^/]+)$', self.output_path).groups()[0]

            subprocess.Popen(['xdg-open', output_path])

    def showvideo(self):
        self.mydialog = QDialog()
        self.mydialog.setModal(True)
        self.mydialog.setWindowTitle("Output")
        self.mydialog.setGeometry(350, 100, 700, 500)
        
        self.mydialog.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        videowidget = QVideoWidget()

        # openBtn = QPushButton('Open Video')
        self.mydialog.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(self.output_path)))
        

        self.mydialog.playBtn = QPushButton()
        self.mydialog.playBtn.setIcon(self.mydialog.style().standardIcon(QStyle.SP_MediaPlay))
        self.mydialog.playBtn.clicked.connect(self.play_video)

        self.mydialog.slider = QSlider(Qt.Horizontal)
        self.mydialog.slider.setRange(0,0)
        self.mydialog.slider.sliderMoved.connect(self.set_position)

        self.mydialog.label = QLabel()
        self.mydialog.label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        hboxLayout = QHBoxLayout()
        hboxLayout.setContentsMargins(0,0,0,0)

        # hboxLayout.addWidget(openBtn)
        hboxLayout.addWidget(self.mydialog.playBtn)
        hboxLayout.addWidget(self.mydialog.slider)

        vboxLayout = QVBoxLayout()
        vboxLayout.addWidget(videowidget)
        vboxLayout.addLayout(hboxLayout)
        vboxLayout.addWidget(self.mydialog.label)

        self.mydialog.setLayout(vboxLayout)
 
        self.mydialog.mediaPlayer.setVideoOutput(videowidget)


        self.mydialog.mediaPlayer.stateChanged.connect(self.mediastate_changed)
        self.mydialog.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mydialog.mediaPlayer.durationChanged.connect(self.duration_changed)

        self.mydialog.exec()

    def play_video(self):
        if self.mydialog.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mydialog.mediaPlayer.pause()
 
        else:
            self.mydialog.mediaPlayer.play()

    def set_position(self, position):
        self.mydialog.mediaPlayer.setPosition(position)

    def mediastate_changed(self, state):
        if self.mydialog.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mydialog.playBtn.setIcon(
                self.mydialog.style().standardIcon(QStyle.SP_MediaPause)
 
            )
 
        else:
            self.mydialog.playBtn.setIcon(
                self.mydialog.style().standardIcon(QStyle.SP_MediaPlay)
 
            )

    def position_changed(self, position):
        self.mydialog.slider.setValue(position)

    def duration_changed(self, duration):
        self.mydialog.slider.setRange(0, duration)

    def set_position(self, position):
        self.mydialog.mediaPlayer.setPosition(position)

    def sleep_btn_process(self):
        self.btn_process.setEnabled(False)
        self.btn_process.setStyleSheet(self.stylesheet_process_inactive)
        self.btn_write.setFont(self.font_asleep)

    def unselect_btn_select(self):
        self.btn_select_video.setStyleSheet(self.stylesheet_select_unselected)
        self.btn_select_video.setText("Select Video")

    def selected_btn_select(self):
        self.btn_select_video.setStyleSheet(self.stylesheet_select_selected)
        video_name = re.search('[^/]*$', self.video_path).group()
        self.btn_select_video.setText(video_name)
    
    def wake_process(self):
        self.btn_process.setEnabled(True)
        self.btn_process.setStyleSheet(self.stylesheet_process_active)
        self.btn_process.setFont(self.font_awake)

    def sleep_btn_process(self):
        self.btn_process.setEnabled(False)
        self.btn_process.setStyleSheet(self.stylesheet_process_inactive)
        self.btn_process.setFont(self.font_asleep)


def main():
    print("Entering fbs")
    appctxt = ApplicationContext()       # 1. Instantiate ApplicationContext
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow, appctxt)
    MainWindow.show()
    exit_code = appctxt.app.exec_()      # 2. Invoke appctxt.app.exec_()
    sys.exit(exit_code)
    os.system('fbs run')


if __name__ == '__main__':
    main()