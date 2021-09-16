import sys 
from PyQt5 import QtWidgets
import app as application

def main():
    print('run main()')
    app = QtWidgets.QApplication(sys.argv)  
    window = application.App() 
    window.show()  # Показываем окно
    sys.exit(app.exec_())
    
if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()