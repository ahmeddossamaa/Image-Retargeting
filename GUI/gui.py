import sys

import socketio
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog, QStackedWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QSizePolicy
)
from PyQt5.QtCore import Qt, QUrl

from config.constants import DataPath
from src.api import retarget_image
import os

# Custom stylesheet for the application
stylesheet = """
QWidget {
    background-color: #F2F4F5;
    color: #000000;
}
QPushButton {
    background-color: #FFA500;
    color: white;
    border-radius: 20px;
    padding: 10px;
    border: 2px solid #FFA500;
    font-family: Changa;
    font-size: 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #FF8C00;
}
QPushButton:pressed {
    background-color: #FF4500;
}

#image_button, #video_button, #about_button {
    background-color: #FFFFFF;
    color: #223C5D;
    border-radius: 20px;
    font-family: Changa;
    font-size: 20px;
}

#image_button:hover, #video_button:hover, #about_button:hover {
    background-color: #F0A90D;
}
#title_label {
    font-size: 34px;
    color: white;
    background-color: #FFFFF;
    border-radius: 10px;
    padding: 5px;
}
#progress_label {
    font-size: 20px;
    color: #000000;
}
#before_label, #after_label {
    font-size: 25px;
    color: #223C5D;
    
}
QSlider::groove:horizontal {
    border: 1px solid #FFFFF;
    height: 15px;
    background: #F3A447;
    border-radius: 7px;
}
QSlider::handle:horizontal {
    background: #F3A447;
    border: 1px solid #F3A447;
    width: 20px;
    height: 20px;
    margin: -3px 0;
    border-radius: 10px;
}
QSlider::sub-page:horizontal {
    background: #F3A447  ;
    border: 1px solid #223C5D;
    height: 15px;
    border-radius: 7px;
}
QSlider::add-page:horizontal {
    background: #223C5D;
    border: 1px solid #F3A447;
    height: 15px;
    border-radius: 7px;
}
QVideoWidget {
    border: 1px solid #005EBD;
}
"""

class MainPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        # Background image
        self.background_label = QLabel(self)
        self.background_label.setPixmap(QPixmap('1.png'))
        self.background_label.setScaledContents(True)  # Ensures the image is scaled to fit the label
        self.background_label.setGeometry(0, 0, 1440, 1050)  # Adjust according to your window size

        # Image button
        self.image_button = QPushButton('Image', self)
        self.image_button.setObjectName("image_button")
        self.image_button.setGeometry(520, 500, 400, 75)  # Set position and size
        self.image_button.clicked.connect(self.show_image_page)

        # Video button
        self.video_button = QPushButton('Video', self)
        self.video_button.setObjectName("video_button")
        self.video_button.setGeometry(520, 630, 400, 75)  # Set position and size
        self.video_button.clicked.connect(self.show_video_page)

        # About button
        self.about_button = QPushButton('About', self)
        self.about_button.setObjectName("about_button")
        self.about_button.setGeometry(520, 760, 400, 75)  # Set position and size
        self.about_button.clicked.connect(self.show_about_page)

    def show_image_page(self):
        self.stacked_widget.setCurrentIndex(1)

    def show_video_page(self):
        self.stacked_widget.setCurrentIndex(2)

    def show_about_page(self):
        self.stacked_widget.setCurrentIndex(3)


class ImagePage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        layout = QVBoxLayout()
        self.background_label = QLabel(self)
        self.background_label.setPixmap(QPixmap('2.png'))
        self.background_label.setScaledContents(True)  # Ensures the image is scaled to fit the label
        self.background_label.setGeometry(0, 0, 1440, 1050)  # Adjust according to your window size

        top_layout = QHBoxLayout()
        self.back_button = QPushButton('Back')
        self.back_button.clicked.connect(self.show_main_page)
        top_layout.addWidget(self.back_button, alignment=Qt.AlignLeft)
        layout.addLayout(top_layout)

        before_after_layout = QHBoxLayout()

        # Before layout
        before_layout = QVBoxLayout()
        self.before_label = QLabel('Before')
        self.before_label.setObjectName("before_label")
        self.before_label.setStyleSheet("background-color: rgba(0, 0, 0, 0);")

        self.before_label.setAlignment(Qt.AlignCenter)
        self.before_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.before_pixmap = QLabel()
        self.before_pixmap.setFixedSize(650, 400)

        self.before_pixmap.setStyleSheet("background-color: rgba(0, 0, 0, 0);")  # Optional: Add border to pixmap
        before_layout.addWidget(self.before_label)
        before_layout.addWidget(self.before_pixmap, alignment=Qt.AlignHCenter)
        before_after_layout.addLayout(before_layout)

        # After layout
        after_layout = QVBoxLayout()
        self.after_label = QLabel('After')
        self.after_label.setObjectName("after_label")
        self.after_label.setStyleSheet("background-color: rgba(0, 0, 0, 0);")

        self.after_label.setAlignment(Qt.AlignCenter)
        self.after_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.after_pixmap = QLabel()
        self.after_pixmap.setFixedSize(650, 400)
        self.after_pixmap.setStyleSheet("background-color: rgba(0, 0, 0, 0);")  # Optional: Add border to pixmap

        after_layout.addWidget(self.after_label)
        after_layout.addWidget(self.after_pixmap, alignment=Qt.AlignHCenter)
        before_after_layout.addLayout(after_layout)

        layout.addLayout(before_after_layout)

        middle_layout = QHBoxLayout()
        self.upload_button = QPushButton('Upload Image')
        self.upload_button.setFixedSize(150, 50)
        self.upload_button.clicked.connect(self.upload_image)
        self.retarget_button = QPushButton('Retarget')
        self.retarget_button.setFixedSize(150, 50)
        self.retarget_button.clicked.connect(self.retarget_image)

        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setObjectName("ratio_slider")
        self.slider.setRange(0, 100)
        self.slider.setValue(75)
        self.slider.valueChanged.connect(self.update_slider_label)
        self.slider.setFixedWidth(300)
        self.slider_label = QLabel("0.75")
        self.slider_label.setStyleSheet("font-family: Changa; font-size: 20px; color: #223C5D; background-color: rgba(0, 0, 0, 0);")
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.slider_label)

        middle_layout.addStretch()
        middle_layout.addWidget(self.upload_button)
        middle_layout.addWidget(self.retarget_button)
        middle_layout.addLayout(slider_layout)
        middle_layout.addStretch()
        layout.addLayout(middle_layout)

        self.setLayout(layout)

    def update_slider_label(self):
        self.slider_label.setText(f"{self.slider.value() / 100:.2f}")

    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Upload Image", "",
                                                   "All Files (*);;Image Files (*.png;*.jpg;*.jpeg)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)

            self.before_pixmap.setPixmap(pixmap)
            print(f"Uploaded image: {file_name}")
            self.current_file_name = file_name

    def retarget_image(self):
        if hasattr(self, 'current_file_name'):
            try:
                ratio = self.slider.value() / 100.0
                retargeted_image = retarget_image(self.current_file_name, ratio=ratio)  # Returns a NumPy array
                height, width, channel = retargeted_image.shape
                bytes_per_line = 3 * width
                q_image = QImage(retargeted_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                q_pixmap = QPixmap.fromImage(q_image)
                self.after_pixmap.setPixmap(q_pixmap)
                print("Retargeting image...")
            except Exception as e:
                print(f"An error occurred while retargeting the image: {e}")
        else:
            print("No image uploaded to retarget.")

    def show_main_page(self):
        self.stacked_widget.setCurrentIndex(0)


class VideoPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.sio = socketio.Client(reconnection=True, reconnection_attempts=3,
                                   reconnection_delay=5, reconnection_delay_max=5, logger=True)
        global frame_counter, frames_count
        frame_counter = 0
        frames_count = 0
        self.stacked_widget = stacked_widget
        self.is_playing = False

        layout = QVBoxLayout()
        self.background_label = QLabel(self)
        self.background_label.setPixmap(QPixmap('3.png'))
        self.background_label.setScaledContents(True)
        self.background_label.setGeometry(0, 0, 1440, 900)

        top_layout = QHBoxLayout()
        self.back_button = QPushButton('Back')
        self.back_button.clicked.connect(self.show_main_page)
        top_layout.addWidget(self.back_button, alignment=Qt.AlignLeft)
        layout.addLayout(top_layout)

        before_after_layout = QHBoxLayout()

        # Before layout
        before_layout = QVBoxLayout()
        self.before_label = QLabel('Before')
        self.before_label.setObjectName("before_label")
        self.before_label.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        self.before_label.setAlignment(Qt.AlignCenter)
        self.before_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.video_player_before = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.video_widget_before = QVideoWidget()
        self.video_player_before.setVideoOutput(self.video_widget_before)
        self.video_widget_before.setFixedSize(650, 400)
        self.video_widget_before.setStyleSheet("border: none; background-color: rgba(255, 255, 255, 255);")

        before_layout.addWidget(self.before_label)
        before_layout.addWidget(self.video_widget_before, alignment=Qt.AlignHCenter)
        before_after_layout.addLayout(before_layout)

        # After layout
        after_layout = QVBoxLayout()
        self.after_label = QLabel('After')
        self.after_label.setObjectName("after_label")
        self.after_label.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        self.after_label.setAlignment(Qt.AlignCenter)
        self.after_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.video_player_after = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.video_widget_after = QVideoWidget()
        self.video_player_after.setVideoOutput(self.video_widget_after)
        self.video_widget_after.setFixedSize(650, 400)
        self.video_widget_after.setStyleSheet("border: none; background-color: rgba(255, 255, 255, 255);")

        after_layout.addWidget(self.after_label)
        after_layout.addWidget(self.video_widget_after, alignment=Qt.AlignHCenter)
        before_after_layout.addLayout(after_layout)

        layout.addLayout(before_after_layout)

        middle_layout = QHBoxLayout()
        self.upload_button = QPushButton('Upload Video')
        self.upload_button.setFixedSize(150, 50)
        self.upload_button.clicked.connect(self.upload_video)
        self.retarget_button = QPushButton('Retarget')
        self.retarget_button.setFixedSize(150, 50)
        self.retarget_button.clicked.connect(self.retarget_video)
        middle_layout.addStretch()
        middle_layout.addWidget(self.upload_button)
        middle_layout.addWidget(self.retarget_button)

        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setObjectName("ratio_slider")
        self.slider.setRange(0, 100)
        self.slider.setValue(75)
        self.slider.valueChanged.connect(self.update_slider_label)
        self.slider.setFixedWidth(300)
        self.slider_label = QLabel("0.75")
        self.slider_label.setStyleSheet("font-family: Changa; font-size: 20px; color: #223C5D; background-color: rgba(0, 0, 0, 0);")
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.slider_label)
        middle_layout.addLayout(slider_layout)

        middle_layout.addStretch()
        layout.addLayout(middle_layout)

        self.progress_label = QLabel("0%")
        self.progress_label.setObjectName("progress_label")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: white;  font-family: Changa;    font-weight: bold;")
        self.progress_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        layout.addWidget(self.progress_label)

        layout.addSpacing(10)

        play_buttons_layout = QHBoxLayout()
        self.play_button = QPushButton('Play Both')
        self.play_button.setObjectName("play_button")
        self.play_button.setFixedSize(150, 50)
        self.play_button.clicked.connect(self.toggle_play)
        play_buttons_layout.addWidget(self.play_button, alignment=Qt.AlignCenter)

        layout.addLayout(play_buttons_layout)

        self.setLayout(layout)
        self.initSocket()

        self.file_name = None

    def update_slider_label(self):
        self.slider_label.setText(f"{self.slider.value() / 100:.2f}")

    def show_main_page(self):
        self.stacked_widget.setCurrentIndex(0)

    def initSocket(self):
        self.sio.connect(url="http://127.0.0.1:5000")
        self.sio.on('video', self.load_retargeted_video, namespace=None)
        self.sio.on('frame', self.updateProgress)

    def updateProgress(self, progress):
        self.progress = progress
        self.progress_label.setText(f"{int(float(self.progress) * 100)}%")

    def upload_video(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Upload Video", "", "All Files (*);;Video Files (*.mp4;*.avi)",
                                                   options=options)
        if file_name:
            media_content = QMediaContent(QUrl.fromLocalFile(file_name))
            self.video_player_before.setMedia(media_content)
            print(f"Uploaded video: {file_name}")

            self.video_player_before.play()
            self.is_playing = False
            self.play_button.setText('Play Both')

            self.file_name = file_name

    def load_retargeted_video(self, output_path):
        print(f"Received video path: {output_path}")
        if output_path:
            abs_output_path = os.path.abspath(output_path)
            media_content = QMediaContent(QUrl.fromLocalFile(abs_output_path))
            self.video_player_after.setMedia(media_content)
            self.video_player_after.play()
            self.is_playing = True
            self.play_button.setText('Stop Both')
            print(f"Loaded retargeted video: {abs_output_path}")
        else:
            print("Failed to load retargeted video.")

    def retarget_video(self):
        if self.file_name is None:
            print("No file name")
            return

        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "..", "data", "output")

        output_path = os.path.join(output_dir, "out.mp4")

        ratio = self.slider.value() / 100.0
        print("Retargeting video...")
        self.sio.emit('video', (
            self.file_name,
            output_path,
            ratio
        ))

    def toggle_play(self):
        if self.progress_label.text() == "100%":
            if self.is_playing:
                self.video_player_before.pause()
                self.video_player_after.pause()
                self.play_button.setText('Play Both')
            else:
                self.video_player_before.play()
                self.video_player_after.play()
                self.play_button.setText('Stop Both')
            self.is_playing = not self.is_playing
        else:
            print("Videos are not fully processed yet.")

    def show_main_page(self):
        self.stacked_widget.setCurrentIndex(0)

    def finish_retargeting(self):
        self.progress_label.setText("100%")
        print("Video retargeting complete")

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

class AboutPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        layout = QVBoxLayout()

        top_layout = QHBoxLayout()
        self.back_button = QPushButton('Back')
        self.back_button.clicked.connect(self.show_main_page)
        top_layout.addWidget(self.back_button, alignment=Qt.AlignLeft)
        layout.addLayout(top_layout)

        # Add a QLabel to serve as a background image placeholder
        self.image_placeholder = QLabel()
        self.image_placeholder.setPixmap(QPixmap('4.png'))  # Set your placeholder image path here
        self.image_placeholder.setScaledContents(True)  # Scale the image to fit the QLabel size
        layout.addWidget(self.image_placeholder)



        self.setLayout(layout)

    def show_main_page(self):
        self.stacked_widget.setCurrentIndex(0)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.stacked_widget = QStackedWidget()

        self.main_page = MainPage(self.stacked_widget)
        self.image_page = ImagePage(self.stacked_widget)
        self.video_page = VideoPage(self.stacked_widget)
        self.about_page = AboutPage(self.stacked_widget)

        self.stacked_widget.addWidget(self.main_page)
        self.stacked_widget.addWidget(self.image_page)
        self.stacked_widget.addWidget(self.video_page)
        self.stacked_widget.addWidget(self.about_page)

        layout = QVBoxLayout()
        layout.addWidget(self.stacked_widget)

        self.setLayout(layout)
        self.setWindowTitle('RetargetMe')
        self.resize(1440, 900)  # Initial size
        self.setMinimumSize(800, 600)  # Minimum size
        self.setWindowIcon(QIcon('company (3).png'))  # Set your icon path here

app = QApplication(sys.argv)
app.setStyleSheet(stylesheet)  # Apply the custom stylesheet
window = MainWindow()
window.show()
sys.exit(app.exec_())
