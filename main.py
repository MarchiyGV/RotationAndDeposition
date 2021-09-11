import sys 
from PyQt5 import QtWidgets
import app as application

def main():
    app = QtWidgets.QApplication(sys.argv)  
    window = application.App() 
    window.show()  # Показываем окно
    sys.exit(app.exec_())
    print('exit')
    
if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()