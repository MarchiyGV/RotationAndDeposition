from PyQt5.QtCore import QModelIndex, Qt, QAbstractTableModel, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QStyledItemDelegate,
    QInputDialog,
    QFileDialog,
    QPushButton,
    QLineEdit,
    QAction
)
from PyQt5 import QtGui
from pandas import DataFrame, read_excel
import re
import os
from numpy import array, nan
import numpy as np

class Settings(QAbstractTableModel):
    
    upd_signal = pyqtSignal(int)
    index_id = 0
    index_name = 1
    index_variableName = 2
    index_value = 3
    index_units = 4
    index_type = 5
    index_group = 6
    index_dependance = 7
    index_comment = 8 
    indexes_visible = [index_name, index_value, index_units]
    headers = ['id', 'Параметр', 'Переменная', 'Значение', 'Единицы \nизмерения', 'Тип', 'Группа', 'Зависимость', 'Комментарий']
         
    def __init__(self, data=[], parent=None):
        super().__init__(parent)
        self.data = array(data, dtype=object)
        for i in range(self.data.shape[0]):
            if self.data[i,self.index_type] == 'bool':
                    if self.data[i, self.index_value] == 'True':
                        self.data[i, self.index_value] = True
                    elif self.data[i, self.index_value] == 'False':
                        self.data[i, self.index_value] = False
                    else: 
                        print('error with type bool')
            for j in range(self.data.shape[1]):
                if self.data[i,j] is nan:
                    self.data[i,j] = ''

    def isVisible(self, raw):
        s = self.data[raw, self.index_dependance]
        if s:
            var, value = s.split('==')
            ind = np.argwhere(self.data[:, self.index_variableName]==var)
            var = self.data[ind, self.index_value].flatten()[0]
            if self.data[ind, self.index_type]=='bool':
                if value=='False':
                    value=False
                else:
                    value=True
            return (var==value)
        return True
    
    def open_file(path):
        df = read_excel(path)
        return Settings(df)

    def save(self, filename):
        df = DataFrame(self.data)
        df.to_excel(filename+'.xlsx', index=False)
        
    def headerData(self, section: int, orientation: Qt.Orientation, role: int):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.headers[section]
            else:
                return str(1+section)

    def columnCount(self, parent=None):
        return self.data.shape[1]

    def rowCount(self, parent=None):
        return self.data.shape[0]
    
    def data(self, index: QModelIndex, role: int):
        if index.isValid():
            if role == Qt.ToolTipRole:
                row=index.row()
                col=index.column()
                if col==self.index_name:
                    comment = self.data[row][self.index_comment]
                    if comment:
                        return str('<p style="width: 100px;">'+comment+'</p>')
                elif col==self.index_value:
                    if self.data[row][self.index_type]=='filename':
                        return self.data[row][col]
            if role == Qt.DisplayRole or role == Qt.EditRole:
                row = index.row()
                col = index.column()
                return str(self.data[row][col])
        
    def wrap(self):
        j = self.index_variableName
        k = self.index_value
        return {self.data[i][j]: self.data[i][k] for i in range(len(self.data))}
    
    def get(self, key):
        settings = self.settigs()
        return settings[key]
        
    def flags(self, index):
        if index.column()==self.index_value:
            return Qt.ItemIsEnabled | Qt.ItemIsEditable
        if index.column()==self.index_name or index.column()==self.index_units:
            return Qt.ItemIsEnabled
        else: 
            return Qt.ItemIsEnabled

    def setData(self, index, value, role):
        if role == Qt.EditRole and value!='':
            i = index.row()
            value, flag = self.suit(i, value)
            if flag:
                self.data[i][self.index_value] = value
                self.upd_signal.emit(i)
                return True
        return False
        
    def suit(self, raw_index, value):
        value_type = self.data[raw_index][self.index_type] 
        flag = True
        if value_type == '+float':
            try: value = float(value)
            except: flag = False
            flag = flag and (value > 0)
        elif value_type == 'float':
            try: value = float(value)
            except: flag = False
        elif value_type == '0+float':
            try: value = float(value)
            except: flag = False
            flag = flag and (value >= 0)
        elif value_type == '+int':
            try: value = int(float(value))
            except: flag = False
            flag = flag and (value > 0)
        elif value_type == '%100':
            try: value = float(value)
            except: flag = False
            flag = flag and (value >= 0) and (value <= 100)
        elif value_type == 'bool':
            if value == 'True':
                value = True
            elif value == 'False':
                value = False
            elif (value == 0 or value == 1): 
                value = bool(value)
            else:
                flag = False
        elif re.match('cases', value_type):
            try:
                value = int(value)
            except:
                try: value = float(value)
                except: pass
        elif value_type == 'filename':
            value = str(value)
            t = re.match('.+\\..+', value)
            if t:
                value = t.group(0)
            else: 
                flag = False
        else:
            print(f'incorrect value {value} ({type(value)})')
            flag = False
        return value, flag 
    
class YesNoDelegate(QStyledItemDelegate):
    def __init__(self, wiget):
        super().__init__(wiget)
        self.labels = ['Да', 'Нет']
        self.items = [True, False]
        
    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItems(self.labels)
        combo.currentIndexChanged.connect(self.currentIndexChanged)
        return combo
        
    def setEditorData(self, editor, index):
        editor.blockSignals(True)
        if index.model().data(index) == 'True':
            i = 0
        elif index.model().data(index) == 'False':
            i = 1
        else:
            print(f'err: {index.model().data(index)}, {type(index.model().data(index))}')
        editor.setCurrentIndex(i)
        editor.blockSignals(False)
        
    def setModelData(self, editor, model, index):
        model.setData(index, self.items[editor.currentIndex()])
        
    @pyqtSlot()
    def currentIndexChanged(self):
        self.commitData.emit(self.sender())
    
    
class DropboxDelegate(QStyledItemDelegate):
    def __init__(self, wiget, items, labels=None):
        super().__init__(wiget)
        self.items = items
        if labels:
            self.labels = labels
        else:
            self.labels = items
        
    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItems(self.labels)
        combo.currentIndexChanged.connect(self.currentIndexChanged)
        return combo
        
    def setEditorData(self, editor, index):
        editor.blockSignals(True)
        editor.setCurrentIndex(self.items.index(index.model().data(index)))
        editor.blockSignals(False)
        
    def setModelData(self, editor, model, index):
        model.setData(index, self.items[editor.currentIndex()])
        
    @pyqtSlot()
    def currentIndexChanged(self):
        self.commitData.emit(self.sender())
        
class BrowseEdit(QLineEdit):
    def __init__(self, contents='', filefilters=None,
        btnicon=None, btnposition=None,
        opendialogtitle=None, opendialogdir=None, parent=None):
        super().__init__(contents, parent)
        self.btnposition = btnposition or QLineEdit.TrailingPosition
        self.reset_action()

    def _clear_actions(self):
        for act_ in self.actions():
            self.removeAction(act_)

    def reset_action(self):
        self._clear_actions()
        self.btnaction = QAction(QtGui.QIcon("open.svg"), '')
        self.btnaction.triggered.connect(self.on_btnaction)
        self.addAction(self.btnaction, self.btnposition)
        
    @pyqtSlot()
    def on_btnaction(self):
        self.delegate.blockSignals(True)
        self.fname = QFileDialog.getOpenFileName(self.parent(), 'Open file', os.getcwd())[0]
        self.delegate.blockSignals(False)
        if not self.fname: return
        self.fname = self.fname.replace('/', os.sep)
        self.setText(self.fname)
        

        
class OpenFileDelegate(QStyledItemDelegate):
    def __init__(self, wiget):
        super().__init__(wiget)
        self.fname = ''
        self.wiget = wiget
        
    def createEditor(self, parent, option, index):
        editor = BrowseEdit(parent=parent)
        editor.delegate = self
        return editor
    
    def setEditorData(self, editor, index):
        editor.blockSignals(True)
        editor.setText(str(index.model().data(index)))
        editor.blockSignals(False)
        
    def setModelData(self, editor, model, index):
        model.setData(index, editor.text())
       
    @pyqtSlot()
    def openFile(self):
        self.commitData.emit(self.sender())
