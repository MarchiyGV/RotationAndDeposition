import sys 
from PyQt5.QtWidgets import QApplication
import app as application
from multiprocessing import freeze_support

def main():
    print('run main()')
    app = QApplication(sys.argv)  
    window = application.App() 
    window.show()  # Показываем окно
    sys.exit(app.exec_())
    
if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()