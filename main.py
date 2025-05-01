import sys, re
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QPainter, QPixmap, QRegion, QPainterPath, QIcon, QFont, QFontMetrics, QColor, QPen
from PyQt5.QtCore import Qt, QTimer, QThread, QRectF, QSize
from PyQt5.QtWidgets import QFrame, QApplication, QMainWindow, QLineEdit, QPushButton, QButtonGroup, QVBoxLayout, QWidget, QLabel, QSizePolicy, QHBoxLayout, QScrollArea, QGraphicsDropShadowEffect

import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

from utils_route_short import *
from utils_route_medium import *
from utils_route_long import *
from utils_path_plan import *
from utils_control_platform import *

def format_json(obj, indent=4):
    if isinstance(obj, dict):
        result = "{\n"
        items = []
        for k, v in obj.items():
            space = ' ' * indent
            if isinstance(v, list):
                items.append(f'{space}"{k}": [{", ".join(json.dumps(x) for x in v)}]')
            else:
                items.append(f'{space}"{k}": {format_json(v, indent + 4)}')
        result += ",\n".join(items)
        result += "\n" + ' ' * (indent - 4) + "}"
        return result
    else:
        return json.dumps(obj)
    
class DummyChat:
    def __init__(self):
        self.system_prompt = ""

class ChatBubble(QWidget):
    def __init__(self, text, is_user, title, max_width):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)

        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)

        title_label = QLabel(title)
        title_label.setObjectName("title")
        title_label.setStyleSheet("font-family: 'Comic Sans MS'; font-size: 30px; color: black;")

        icon_label = QLabel()
        icon_label.setFixedSize(40, 40)

        if is_user:
            icon_label.setPixmap(QtGui.QPixmap('icons/user_icon.png').scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            title_layout.addStretch()
            title_layout.addWidget(title_label)
            title_layout.addWidget(icon_label)
        else:
            icon_label.setPixmap(QtGui.QPixmap('icons/ai_icon.png').scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            title_layout.addWidget(icon_label)
            title_layout.addWidget(title_label)
            title_layout.addStretch()

        bubble_layout = QHBoxLayout()
        bubble_layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel()
        label.setText(text)

        label.setMaximumWidth(max_width)
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setTextFormat(Qt.PlainText)

        if is_user:
            label.setStyleSheet("font-family: 'Comic Sans MS'; font-size: 24px; background-color: rgb(224, 244, 218); border-radius: 15px; color: black; border: 2px solid rgb(53, 105, 32); padding: 8px 16px;")
            label.setAlignment(Qt.AlignJustify)
            bubble_layout.addStretch()
            bubble_layout.addWidget(label)
        else:
            label.setStyleSheet("font-family: 'Comic Sans MS'; font-size: 24px; background-color: rgb(224, 224, 224); border-radius: 15px; color: black; border: 2px solid black; padding: 8px 16px;")
            label.setAlignment(Qt.AlignJustify)
            bubble_layout.addWidget(label)
            bubble_layout.addStretch()

        outer_layout.addLayout(title_layout)
        outer_layout.addLayout(bubble_layout)
        self.setLayout(outer_layout)

        self.label = label
        
class ChatWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.text = "NeLV" 
        self.opacity = 0.5
        self.text_width = 400
        self.text_height = 300

    def setOpacity(self, opacity):
        self.opacity = opacity
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.white)
        painter.setOpacity(self.opacity)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(Qt.gray); pen.setWidth(1); painter.setPen(pen); painter.setPen(QColor(255, 231, 163)) 
        font = QFont("Comic Sans MS", 60); font.setBold(True); painter.setFont(font)
        fm = QFontMetrics(font)
        text_width = fm.horizontalAdvance(self.text)
        text_height = fm.height()
        x = (self.width() - text_width) // 2
        y = (self.height() + text_height) // 2 - fm.descent()
        painter.drawText(x, y, self.text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        path = QPainterPath()
        rectf = QRectF(self.rect())
        path.addRoundedRect(rectf, 15, 15)
        region = QRegion(path.toFillPolygon().toPolygon())
        self.setMask(region)

class FluentButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.setCheckable(True)
        self.setAutoExclusive(True) 
        self.toggled.connect(self.update)
        self.setFixedSize(50, 50)
        self._line_color = QColor(0, 124, 194)
        self._line_width = 5
        self._line_height = 20
        self.setStyleSheet(""" QPushButton { font-family: 'Segoe UI Emoji'; font-size: 18px; border: none; border-radius: 10px; background-color: transparent; }
                           QPushButton:hover { background-color: rgb(221, 227, 233); }
                           QPushButton:checked { background-color: rgb(221, 227, 233); } """)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.isChecked():
            painter = QPainter(self); painter.setRenderHint(QPainter.Antialiasing); painter.setBrush(self._line_color);  painter.setPen(Qt.NoPen)
            x = 0; y = int((self.height() - self._line_height) / 2)
            w = int(self._line_width); h = int(self._line_height)
            painter.drawRoundedRect(x, y, w, h, w/2, w/2)

class ChatbotApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLMAP")
        self.setWindowIcon(QIcon("icons/title_icon.png"))
        self.setGeometry(1200, 600, 1000, 1150)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.setStyleSheet(""" QMainWindow { background-color: #d3e3fd }
                           QScrollBar { background: #e0e0e0; border-radius: 5px; margin: 0; }
                           QScrollBar:vertical { width: 10px; } QScrollBar:horizontal { height: 10px; }
                           QScrollBar::handle { background: #9e9e9e; border-radius: 5px; min-height: 30px; min-width: 30px; }
                           QScrollBar::handle:hover { background: #757575; }
                           QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page { background: none; height: 0; width: 0; }""")
        
        main_layout = QHBoxLayout()
        
        self.chat_widget = ChatWidget(self)
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_layout.setSpacing(10)
        self.chat_layout.setAlignment(Qt.AlignTop)

        self.chat_container = QWidget(); self.container_layout = QVBoxLayout(self.chat_container); self.container_layout.setContentsMargins(0, 0, 0, 0); self.container_layout.setSpacing(0)

        self.chat_title = "Chat"
        self.chat_title_area = QWidget()
        self.chat_title_area.setStyleSheet("background-color: transparent; padding: 0px")
        self.chat_title_label = QLabel(self.chat_title)
        self.chat_title_label.setAlignment(Qt.AlignCenter)
        self.chat_title_label.setStyleSheet("font-family: 'Comic Sans MS'; font-size: 14pt; font-weight: bold;")
        self.chat_title_layout = QHBoxLayout(self.chat_title_area)
        self.chat_title_layout.addWidget(self.chat_title_label)

        self.chat_scroll_area = QScrollArea()
        self.chat_scroll_area.setWidgetResizable(True)
        self.chat_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.chat_scroll_area.setWidget(self.chat_widget)
        self.chat_scroll_area.setViewportMargins(0, 0, 10, 0)
        self.chat_scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.container_layout.addWidget(self.chat_title_area)
        self.container_layout.addWidget(self.chat_scroll_area)

        self.nav_bar = QWidget()
        self.nav_bar.setFixedWidth(60)
        self.nav_bar.setStyleSheet(" background-color: white; border-radius: 15px; ")
        nav_layout = QVBoxLayout(self.nav_bar)
        nav_layout.setAlignment(Qt.AlignTop)
        nav_layout.setContentsMargins(5, 10, 5, 10)
        nav_layout.setSpacing(10)
        
        self.nav_symbols = [("💬", "Chat"), ("🚁", "Short Range"), ("🛩️", "Medium Range"), ("✈️", "Long Range"),
                            ("✏️", "Plan Route"), ("🗺️", "Plan Path"), ("🕹️", "Upload Path"), 
                            ("📋", "Hist - Short Range"), ("📋", "Hist - Medium Range"), ("📋", "Hist - Long Range"), 
                            ("🔄", "Reset"), ]
        
        
        button_group = QButtonGroup()
        button_group.setExclusive(True)
        button_group.buttonClicked.connect(lambda button: self.choose_model(button_group.id(button)))
        for i, (symbol, tooltip) in enumerate(self.nav_symbols):
            if i == 0:
                label = QLabel("Chat"); label.setAlignment(Qt.AlignCenter); label.setStyleSheet("font-family: 'Comic Sans MS'; font-size: 18px; color: black; padding-top: 10px; padding-bottom: 0px;")
                nav_layout.addWidget(label)
            elif i == 1:
                separator = QWidget(); separator.setFixedHeight(1); separator.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed); separator.setStyleSheet("background-color: rgba(0, 0, 0, 50);")
                nav_layout.addWidget(separator)
                label = QLabel("Mode"); label.setAlignment(Qt.AlignCenter); label.setStyleSheet("font-family: 'Comic Sans MS'; font-size: 18px; color: black; padding-top: 10px; padding-bottom: 0px;")
                nav_layout.addWidget(label)
            elif i == 4:
                separator = QWidget(); separator.setFixedHeight(1); separator.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed); separator.setStyleSheet("background-color: rgba(0, 0, 0, 50);")
                nav_layout.addWidget(separator)
                label = QLabel("Tool"); label.setAlignment(Qt.AlignCenter); label.setStyleSheet("font-family: 'Comic Sans MS'; font-size: 18px; color: black; padding-top: 10px; padding-bottom: 0px;")
                nav_layout.addWidget(label)
            elif i == 7:
                separator = QWidget(); separator.setFixedHeight(1); separator.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed); separator.setStyleSheet("background-color: rgba(0, 0, 0, 50);")
                nav_layout.addWidget(separator)
                label = QLabel("Hist"); label.setAlignment(Qt.AlignCenter); label.setStyleSheet("font-family: 'Comic Sans MS'; font-size: 18px; color: black; padding-top: 10px; padding-bottom: 0px;")
                nav_layout.addWidget(label)

            if tooltip in ["Plan Route", "Plan Path", "Upload Path", "Reset"]:
                button = QPushButton(symbol); button.setFixedSize(50, 50)
                if tooltip in ["Reset"]:
                    nav_layout.addStretch()
                    button.setStyleSheet(""" QPushButton { font-family: 'Segoe UI Emoji'; font-size: 18px; border: none; border-radius: 10px; background-color: transparent; }
                                        QPushButton:hover { background-color: rgb(240, 128, 128); } 
                                        QPushButton:pressed { background-color: rgb(220, 20, 60); } """)
                elif tooltip in ["Plan Route", "Plan Path", "Upload Path"]:
                    button.setStyleSheet(""" QPushButton { font-family: 'Segoe UI Emoji'; font-size: 18px; border: none; border-radius: 10px; background-color: transparent; }
                                        QPushButton:hover { background-color: rgb(255, 255, 153); } 
                                        QPushButton:pressed { background-color: rgb(255, 215, 0); } """)
                    
            else:
                button = FluentButton(symbol)
            button.setToolTip(tooltip)
            button_group.addButton(button, i)
            nav_layout.addWidget(button)
            if i == 0:
                button.setChecked(True)
                self.mode = "Chat"
                self.route_planner = DummyChat()

            
        main_layout.addWidget(self.nav_bar)
        main_layout.addWidget(self.chat_container)


        input_layout = QHBoxLayout()
        
        self.text_input = QLineEdit(self)
        self.text_input.setPlaceholderText("Where are we going today?")
        self.text_input.setStyleSheet("font-family: 'Comic Sans MS'; font-size: 30px; background-color: white; border-radius: 15px; color: rgb(93, 93, 93); border: 1px solid gray; padding: 8px 16px;")
        shadow = QGraphicsDropShadowEffect(self.text_input); shadow.setBlurRadius(15); shadow.setOffset(0, 3); shadow.setColor(QColor(0, 0, 0, 160))
        self.text_input.setGraphicsEffect(shadow)
        input_layout.addWidget(self.text_input)

        self.send_button = QPushButton(self)
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setIcon(QIcon("icons/send_icon.png"))
        self.send_button.setIconSize(QSize(30, 30))
        self.send_button.setFixedSize(60, 60)
        self.send_button.setStyleSheet("background-color: black; border-radius: 30px; ")
        input_layout.addWidget(self.send_button)

        final_layout = QVBoxLayout()
        final_layout.addLayout(main_layout)
        final_layout.addLayout(input_layout)
        central_widget.setLayout(final_layout)
        
        self.messages = [{"role": "system", "content": self.route_planner.system_prompt}]
        self.n_generation = 10
    
    def choose_model(self, button_id):
        self.mode = self.nav_symbols[button_id][1]
        self.chat_title = self.mode if self.mode in ["Chat", "Short Range", "Medium Range", "Long Range"] else self.chat_title
        self.chat_title = self.mode[7:] if "Hist" in self.mode else self.chat_title
        self.chat_title_label.setText(self.chat_title)
        print(self.mode)
    
        if self.mode == "Chat":
            self.route_planner = DummyChat()
            self.reset_chat()
        elif self.mode == "Short Range":
            self.route_planner = ShortRoutePlanner()
            self.reset_chat()
        elif self.mode == "Medium Range":
            self.route_planner = MediumRoutePlanner()
            self.reset_chat()
        elif self.mode == "Long Range":
            self.route_planner = LongRoutePlanner()
            self.reset_chat()

        elif self.mode == "Plan Route":
            if not isinstance(self.route_planner, DummyChat):
                self.route_planner.plan_route(self.planned_flight)
                route_plot = QLabel(); route_pixmap = QtGui.QPixmap('./temp/fig_route.png').scaled(800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation); route_plot.setPixmap(route_pixmap); route_plot.setFixedSize(route_pixmap.size()) 
                self.chat_layout.addWidget(route_plot)
        elif self.mode == "Plan Path":
            if not isinstance(self.route_planner, DummyChat):
                self.path_planner = Path_Planner()
                self.path_planner.plan_path("temp/route_coordinates.txt", self.n_generation, self.route_planner.xylims, self.route_planner.map_source)
                route_plot = QLabel(); route_pixmap = QtGui.QPixmap('./temp/fig_path.png').scaled(800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation); route_plot.setPixmap(route_pixmap); route_plot.setFixedSize(route_pixmap.size()) 
                self.chat_layout.addWidget(route_plot)
        elif self.mode == "Upload Path":
            if not isinstance(self.route_planner, DummyChat):
                execute_pipeline()

        elif self.mode == "Hist - Short Range":
            self.route_planner = ShortRoutePlanner()
            self.reset_chat()
            with open("examples/short_range.json", "r", encoding="utf-8") as file:
                self.messages = json.load(file)
                self.reconstruction_hist()
        elif self.mode == "Hist - Medium Range":
            self.route_planner = MediumRoutePlanner()
            self.reset_chat()
            with open("examples/medium_range.json", "r", encoding="utf-8") as file:
                self.messages = json.load(file)
                self.reconstruction_hist()
        elif self.mode == "Hist - Long Range":
            self.route_planner = LongRoutePlanner()
            self.reset_chat()
            with open("examples/long_range.json", "r", encoding="utf-8") as file:
                self.messages = json.load(file)
                self.reconstruction_hist()
        elif self.mode == "Reset":
            self.reset_chat()
                
    def reconstruction_hist(self):
        window_width = self.width()
        max_bubble_width = int(window_width)
        for message in self.messages:
            content = message['content']
            if message['role'] == 'user':
                if "\nOnly output the JSON object. No prefix, additional text, or explanation.\n\n" in content:
                    content = content.replace("\nOnly output the JSON object. No prefix, additional text, or explanation.\n\n", "")
                bubble = ChatBubble(content, True, "User", max_bubble_width)
                self.chat_layout.addWidget(bubble)
            elif message['role'] == 'assistant':
                parsed_json = json.loads(content)
                bubble = ChatBubble(format_json(parsed_json), False, "LLM-as-Parser", max_bubble_width)
                self.chat_layout.addWidget(bubble)
        self.load_response(message['content'])
                
    def reset_chat(self):
        while self.chat_layout.count():
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.messages = [{"role": "system", "content": self.route_planner.system_prompt}]

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
            self.send_button.click()

    def generate_response(self, message, LLM_model, LLM_tokenizer):
        if self.mode in ["Short Range", "Medium Range", "Long Range"]:
            self.messages.append({"role": "user", "content": f"{message}\nOnly output the JSON object. No prefix, additional text, or explanation.\n\n"})
        else:
            self.messages.append({"role": "user", "content": message})
        pipe = pipeline("text-generation", model=LLM_model, tokenizer=LLM_tokenizer,)
        generation_args = {"max_new_tokens": 512, "return_full_text": False, "temperature": 0.0, "do_sample": False, }
        response = pipe(self.messages, **generation_args)[0]['generated_text']
        # self.messages.append({"role": "assistant", "content": f"{response.strip().rstrip("\n")}\n\n"})
        return response
    
    def load_response(self, response):
        try:
            self.planned_flight = json.loads(response)
        except:
            try:
                if "{" in response and "}" in response:
                    last_open = response.rindex("{")
                    last_close = response.rindex("}")
                    response = response[last_open:last_close+1]
                try:
                    self.planned_flight = json.loads(response)
                except:
                    print("=" * 50)
                    print("Wrong JSON format!")
                    print(response)
                    print("=" * 50)
                    self.planned_flight = self.route_planner.default_flight
            except:
                print("=" * 50)
                print("Wrong JSON format!")
                print(response)
                print("=" * 50)
                self.planned_flight = self.route_planner.default_flight

    def send_message(self):
        window_width = self.width()
        max_bubble_width = int(window_width)
        user_input = self.text_input.text().strip()

        u_msg = ChatBubble(user_input, True, "User", max_bubble_width)
        self.chat_layout.addWidget(u_msg)
        # response = self.generate_response(user_input, LLM_model, LLM_tokenizer)
        response = {}
        
        if self.mode != "Chat":
            self.load_response(response)
            llm_msg = ChatBubble(format_json(self.planned_flight), False, "LLM-as-Parser", max_bubble_width)
        else:
            llm_msg = ChatBubble(response, False, "LLM", max_bubble_width)
        self.chat_layout.addWidget(llm_msg)
        self.text_input.clear()
            


if __name__ == "__main__":

    # LLM_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-4-mini-instruct", device_map="auto", torch_dtype="auto", trust_remote_code=False)
    # LLM_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")

    app = QApplication(sys.argv)
    chatbot_app = ChatbotApp()
    chatbot_app.show()
    sys.exit(app.exec_())