import sys  # sys нужен для передачи argv в QApplication
from PyQt5 import QtWidgets
import design
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
import numpy as np
import functions

class App(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py
        super().__init__()
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        self.DepositionButton.clicked.connect(self.deposition)
        self.update_model()
        self.settings_input.setRowCount(1)
        self.settings_input.setItem(0,0,  QtWidgets.QTableWidgetItem('1231'))
        
    def plot(self, layout, wiget, fig):
        if layout.isEmpty():
            self.canvas = FigureCanvas(fig)
            layout.addWidget(self.canvas)
        else:
            layout.removeWidget(self.canvas)
            layout.removeWidget(self.toolbar)
            self.canvas = FigureCanvas(fig)
            layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, 
                wiget, coordinates=True)
        layout.addWidget(self.toolbar)
        self.canvas.draw()
        
    def update_model(self):
        self.model = functions.Model()
        fig = self.model.plot_mesh()
        self.plot(self.mesh_plot_vl, self.mesh_plot, fig)
        
    def deposition(self):
        I, heterogeneity, I_err = self.model.deposition(15, 1.5, 1, 3)
        fig = Figure()
        ax1f = fig.add_subplot(111)
        ax1f.contourf(self.model.substrate_coords_map_x, 
                      self.model.substrate_coords_map_y, I/I.max())
        #fig.clim(I.min()/I.max(), 1)
        #fig.colorbar(ax1f)
        ax1f.set_xlabel('x, mm')
        ax1f.set_ylabel('y, mm')
        ax1f.set_title(f'Film heterogeneity $H = {round(heterogeneity,2)}\\%$')
        self.plot(self.film_vl, self.film, fig)
    

def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = App()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    #app.exec()  # и запускаем приложение
    #sys.exit()
    sys.exit(app.exec())
    print('exit')
if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()