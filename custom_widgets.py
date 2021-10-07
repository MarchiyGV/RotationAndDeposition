from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QLineEdit, QTableView
from PyQt5.QtGui import QFocusEvent, QKeyEvent

class MyLineEdit(QLineEdit):
    
    focused = pyqtSignal()
    unfocused = pyqtSignal()
    
    def __init__(self, *args, **kwargs):
        super(MyLineEdit, self).__init__(*args, **kwargs) #call to superclass        
    
    @pyqtSlot(QFocusEvent)
    def focusInEvent(self, event):
        QLineEdit.focusInEvent(self, event)
        self.focused.emit()
    
    @pyqtSlot(QFocusEvent)
    def focusOutEvent(self, event):
        QLineEdit.focusOutEvent(self, event)
        self.unfocused.emit()
    
    @pyqtSlot(QKeyEvent)
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
            self.clearFocus()
        QLineEdit.keyPressEvent(self, event)
        
class MyTableView(QTableView):
    
    enter_pressed = pyqtSignal(int)
    
    def __init__(self, *args, **kwargs):
        super(MyTableView, self).__init__(*args, **kwargs) #call to superclass        
    
    @pyqtSlot(QKeyEvent)
    def keyPressEvent(self, event):
        QTableView.keyPressEvent(self, event)
        if event.key() == Qt.Key_Return:
            row = self.currentIndex().row()
            self.enter_pressed.emit(row)