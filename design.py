# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\Георгий\Desktop\ФТИ\RotationAndDeposition\gui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1349, 827)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.InputWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.InputWidget.setMinimumSize(QtCore.QSize(851, 631))
        self.InputWidget.setObjectName("InputWidget")
        self.Model = QtWidgets.QWidget()
        self.Model.setObjectName("Model")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.Model)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.widget_3 = QtWidgets.QWidget(self.Model)
        self.widget_3.setObjectName("widget_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget_3)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.table_settings = QtWidgets.QTableView(self.widget_3)
        self.table_settings.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.table_settings.sizePolicy().hasHeightForWidth())
        self.table_settings.setSizePolicy(sizePolicy)
        self.table_settings.setMinimumSize(QtCore.QSize(400, 530))
        self.table_settings.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.table_settings.setEditTriggers(QtWidgets.QAbstractItemView.AnyKeyPressed|QtWidgets.QAbstractItemView.DoubleClicked|QtWidgets.QAbstractItemView.EditKeyPressed|QtWidgets.QAbstractItemView.SelectedClicked)
        self.table_settings.setAlternatingRowColors(False)
        self.table_settings.setObjectName("table_settings")
        self.verticalLayout_2.addWidget(self.table_settings)
        self.open_settings_Button = QtWidgets.QPushButton(self.widget_3)
        self.open_settings_Button.setObjectName("open_settings_Button")
        self.verticalLayout_2.addWidget(self.open_settings_Button)
        self.save_settings_Button = QtWidgets.QPushButton(self.widget_3)
        self.save_settings_Button.setObjectName("save_settings_Button")
        self.verticalLayout_2.addWidget(self.save_settings_Button)
        self.update_model_Button = QtWidgets.QPushButton(self.widget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.update_model_Button.sizePolicy().hasHeightForWidth())
        self.update_model_Button.setSizePolicy(sizePolicy)
        self.update_model_Button.setObjectName("update_model_Button")
        self.verticalLayout_2.addWidget(self.update_model_Button)
        self.horizontalLayout_2.addWidget(self.widget_3)
        self.widget_2 = QtWidgets.QWidget(self.Model)
        self.widget_2.setObjectName("widget_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.mesh_plot = QtWidgets.QWidget(self.widget_2)
        self.mesh_plot.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mesh_plot.sizePolicy().hasHeightForWidth())
        self.mesh_plot.setSizePolicy(sizePolicy)
        self.mesh_plot.setMinimumSize(QtCore.QSize(300, 300))
        self.mesh_plot.setMaximumSize(QtCore.QSize(600, 600))
        self.mesh_plot.setObjectName("mesh_plot")
        self.mesh_plot_vl = QtWidgets.QVBoxLayout(self.mesh_plot)
        self.mesh_plot_vl.setObjectName("mesh_plot_vl")
        self.verticalLayout.addWidget(self.mesh_plot)
        self.source_plot = QtWidgets.QWidget(self.widget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.source_plot.sizePolicy().hasHeightForWidth())
        self.source_plot.setSizePolicy(sizePolicy)
        self.source_plot.setMinimumSize(QtCore.QSize(300, 300))
        self.source_plot.setMaximumSize(QtCore.QSize(600, 600))
        self.source_plot.setObjectName("source_plot")
        self.source_plot_vl = QtWidgets.QVBoxLayout(self.source_plot)
        self.source_plot_vl.setObjectName("source_plot_vl")
        self.verticalLayout.addWidget(self.source_plot)
        self.horizontalLayout_2.addWidget(self.widget_2)
        self.InputWidget.addTab(self.Model, "")
        self.Deposition = QtWidgets.QWidget()
        self.Deposition.setObjectName("Deposition")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.Deposition)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.film_plot = QtWidgets.QWidget(self.Deposition)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.film_plot.sizePolicy().hasHeightForWidth())
        self.film_plot.setSizePolicy(sizePolicy)
        self.film_plot.setMinimumSize(QtCore.QSize(400, 400))
        self.film_plot.setObjectName("film_plot")
        self.film_vl = QtWidgets.QVBoxLayout(self.film_plot)
        self.film_vl.setObjectName("film_vl")
        self.gridLayout_3.addWidget(self.film_plot, 0, 2, 1, 1)
        self.widget = QtWidgets.QWidget(self.Deposition)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setMinimumSize(QtCore.QSize(400, 300))
        self.widget.setObjectName("widget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.widget_4 = QtWidgets.QWidget(self.widget)
        self.widget_4.setMinimumSize(QtCore.QSize(150, 30))
        self.widget_4.setMaximumSize(QtCore.QSize(200, 50))
        self.widget_4.setObjectName("widget_4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget_4)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_4 = QtWidgets.QLabel(self.widget_4)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_3.addWidget(self.label_4)
        self.thick_edit = QtWidgets.QLineEdit(self.widget_4)
        self.thick_edit.setMaximumSize(QtCore.QSize(60, 16777215))
        self.thick_edit.setObjectName("thick_edit")
        self.horizontalLayout_3.addWidget(self.thick_edit)
        self.label_5 = QtWidgets.QLabel(self.widget_4)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_3.addWidget(self.label_5)
        self.verticalLayout_3.addWidget(self.widget_4)
        self.widget_5 = QtWidgets.QWidget(self.widget)
        self.widget_5.setObjectName("widget_5")
        self.gridLayout = QtWidgets.QGridLayout(self.widget_5)
        self.gridLayout.setObjectName("gridLayout")
        self.NR_disp = QtWidgets.QDoubleSpinBox(self.widget_5)
        self.NR_disp.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.NR_disp.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        self.NR_disp.setDecimals(0)
        self.NR_disp.setSingleStep(1.0)
        self.NR_disp.setObjectName("NR_disp")
        self.gridLayout.addWidget(self.NR_disp, 2, 3, 1, 1)
        self.NR_Slider = QtWidgets.QSlider(self.widget_5)
        self.NR_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.NR_Slider.setObjectName("NR_Slider")
        self.gridLayout.addWidget(self.NR_Slider, 2, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.widget_5)
        self.label_2.setMaximumSize(QtCore.QSize(30, 16777215))
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.widget_5)
        self.label.setMaximumSize(QtCore.QSize(30, 16777215))
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.widget_5)
        self.label_3.setMaximumSize(QtCore.QSize(30, 16777215))
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 2)
        self.R_Slider = QtWidgets.QSlider(self.widget_5)
        self.R_Slider.setMinimumSize(QtCore.QSize(300, 0))
        self.R_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.R_Slider.setObjectName("R_Slider")
        self.gridLayout.addWidget(self.R_Slider, 0, 1, 1, 2)
        self.k_disp = QtWidgets.QDoubleSpinBox(self.widget_5)
        self.k_disp.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.k_disp.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        self.k_disp.setDecimals(0)
        self.k_disp.setSingleStep(1.0)
        self.k_disp.setObjectName("k_disp")
        self.gridLayout.addWidget(self.k_disp, 1, 3, 1, 1)
        self.R_disp = QtWidgets.QDoubleSpinBox(self.widget_5)
        self.R_disp.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.R_disp.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        self.R_disp.setDecimals(0)
        self.R_disp.setSingleStep(1.0)
        self.R_disp.setObjectName("R_disp")
        self.gridLayout.addWidget(self.R_disp, 0, 3, 1, 1)
        self.k_Slider = QtWidgets.QSlider(self.widget_5)
        self.k_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.k_Slider.setObjectName("k_Slider")
        self.gridLayout.addWidget(self.k_Slider, 1, 1, 1, 2)
        self.verticalLayout_3.addWidget(self.widget_5)
        self.DepositionButton = QtWidgets.QPushButton(self.widget)
        self.DepositionButton.setObjectName("DepositionButton")
        self.verticalLayout_3.addWidget(self.DepositionButton)
        self.gridLayout_3.addWidget(self.widget, 0, 1, 1, 1)
        self.InputWidget.addTab(self.Deposition, "")
        self.horizontalLayout.addWidget(self.InputWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1349, 31))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionenter = QtWidgets.QAction(MainWindow)
        self.actionenter.setObjectName("actionenter")

        self.retranslateUi(MainWindow)
        self.InputWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Thickness calculator"))
        self.open_settings_Button.setText(_translate("MainWindow", "Загрузить настройки"))
        self.save_settings_Button.setText(_translate("MainWindow", "Сохранить настройки"))
        self.update_model_Button.setText(_translate("MainWindow", "Обновить модель"))
        self.InputWidget.setTabText(self.InputWidget.indexOf(self.Model), _translate("MainWindow", "Модель"))
        self.label_4.setText(_translate("MainWindow", "Толщина"))
        self.label_5.setText(_translate("MainWindow", "nm"))
        self.label_2.setText(_translate("MainWindow", "k"))
        self.label.setText(_translate("MainWindow", "R"))
        self.label_3.setText(_translate("MainWindow", "NR"))
        self.DepositionButton.setText(_translate("MainWindow", "Напылить"))
        self.InputWidget.setTabText(self.InputWidget.indexOf(self.Deposition), _translate("MainWindow", "Напыление"))
        self.actionenter.setText(_translate("MainWindow", "enter"))
        self.actionenter.setShortcut(_translate("MainWindow", "Return"))

