import sys  # sys нужен для передачи argv в QApplication
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QModelIndex, Qt
import design
import matplotlib
matplotlib.use('QT5Agg')
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
import numpy as np
import functions 
#pyuic5 "C:\Users\Георгий\Desktop\ФТИ\RotationAndDeposition\gui.ui" -o "C:\Users\Георгий\Desktop\ФТИ\RotationAndDeposition/design.py"

settings = [['Путь', 'filename', 'depz.txt', 'filename', 'path to dep profile'],
            ['Скорость осаждения', 'C', 4.46, '+float', 'thickness [nm] per minute'],
            ['source', 'source', 0, [0, 1], 'Choose source of get thickness data 1 - seimtra, 0 - experiment'],
            ['val', 'val', 3, [1, 2, 3], '1, 2, 3 - magnetron position'],
            ['Длина подложки', 'substrate_x_len', 100, '+float', 'Substrate width, mm'],
            ['Ширина подложки', 'substrate_y_len', 100, '+float', 'Substrate length, mm'],
            ['Разрешение по х', 'substrate_x_res', 0.05, '+float', 'Substrate x resolution, 1/mm'],
            ['Разрешение по у', 'substrate_y_res', 0.05, '+float', 'Substrate y resolution, 1/mm'],
            ['Число ядер', 'cores', 1, '+int', 'number of jobs for paralleling'],
            ['Подробный лог', 'verbose', True, [True, False], 'True: print message each time when function of deposition called'],
            ['Стирать кэш', 'delete_cache', True, [True, False], 'True: delete history of function evaluations in the beggining of work. Warning: if = False, some changes in the code may be ignored'],
            ['Точность в точке', 'point_tolerance', 5/100, '+float', 'needed relative tolerance for thickness in each point'],
            ['Максимальный шаг по углу', 'max_angle_divisions', 10, '+int', 'limit of da while integration = 1 degree / max_angle_divisions'],
            ['holder_inner_radius', 'holder_inner_radius', 20, '+float', 'mm'],
            ['holder_outer_radius', 'holder_outer_radius', 145, '+float', 'mm'],
            ['deposition_len_x', 'deposition_len_x', 290, '+float', 'mm'],
            ['deposition_len_y', 'deposition_len_y', 290, '+float', 'mm'],
            ['Разрешение по х источника', 'deposition_res_x', 1, '+float', '1/mm'],
            ['Разрешение по у источника', 'deposition_res_y', 1, '+float', '1/mm'],
            ['Границы R', 'R_bounds', (10, 70), ('+float','+float') , '(min, max) mm'],
            ['Границы k', 'k_bounds', (1, 50), ('+float','+float'), '(min, max)'],
            ['Границы числа поворотов', 'NR_bounds', (1, 100), ('+float','+float'), ''],
            ['Начальное приближение', 'x0', [35, 4.1, 1], ('+float','+float', '+float'),'initial guess for optimisation [R0, k0]'],
            ['Алгоритм минимизации', 'minimizer', 'NM_custom', ['NM_custom', 'NM', 'Powell'], ''],
            ['Средний МК шаг по R', 'R_mc_interval', 5/100, '%100', 'step for MC <= R_mc_interval*(R_max_bound-R_min_bound)'],
            ['Средний МК шаг по k', 'k_mc_interval', 5/100, '%100', 'step for MC <= k_mc_interval*(k_max_bound-k_min_bound)'],
            ['Средний МК шаг по числу оборотов', 'NR_mc_interval', 15/100, '%100', ''],
            ['Мин. МК шаг по R', 'R_min_step', 1, '0+float', 'step for MC >= R_min_step'],
            ['Мин. МК шаг по k', 'k_min_step', 0.01, '0+float', 'step for MC >= k_min_step'],
            ['Мин. МК шаг по числу оборотов', 'NR_min_step', 1, '0+float', ''],
            ['Число МК итераций', 'mc_iter', 2, '+int', 'number of Monte-Carlo algoritm"s iterations (number of visited local minima)'],
            ['МК температура', 'T', 2, '+float', '"temperature" for MC algoritm']]

class Settings(QtCore.QAbstractTableModel):
    def __init__(self, data=[], parent=None):
        super().__init__(parent)
        self.data = data
        self.index_name = 0
        self.index_variableName = 1
        self.index_value = 2
        self.index_type = 3
        self.index_comment = 4 
        self.headers = ['Параметр', 'Значение']

    def headerData(self, section: int, orientation: Qt.Orientation, role: int):
        if role == QtCore.Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.headers[section]
            else:
                return str(1+section)

    def columnCount(self, parent=None):
        return 2

    def rowCount(self, parent=None):
        return len(self.data)
    
    def data(self, index: QModelIndex, role: int):
        if role == QtCore.Qt.DisplayRole:
            row = index.row()
            if index.column() == 0:
                col = self.index_name
            elif index.column() == 1:
                col = self.index_value
            return str(self.data[row][col])
        
    def settigs(self):
        j = self.index_variableName
        k = self.index_value
        return {self.data[i][j]: self.data[i][k] for i in range(len(self.data))}
    
    def get(self, key):
        settings = self.settigs()
        return settings[key]
        
    def flags(self, index):
        if index.column()==1:
            return Qt.ItemIsEnabled | Qt.ItemIsEditable
        if index.column()==0:
            return Qt.ItemIsEnabled

    def setData(self, index, value, role):
        if role == Qt.EditRole and value!='':
            i = index.row()
            value, flag = self.suit(i, value)
            if flag:
                self.data[i][self.index_value] = value
            return True
        return False
        
    def suit(self, raw_index, value):
        value_type = self.data[raw_index][self.index_type] 
        flag = True
        if value_type == '+float':
            try: value = float(value)
            except: flag = False
            flag = flag and (value > 0)
            return value, flag
        if value_type == '0+float':
            try: value = float(value)
            except: flag = False
            flag = flag and (value >= 0)
            return value, flag
        if value_type == '+int':
            print(value)
            try: value = int(float(value))
            except: flag = False
            flag = flag and (value > 0)
            print(flag)
            return value, flag
        if value_type == '%100':
            try: value = float(value)
            except: flag = False
            flag = flag and (value >= 0) and (value <= 1)
            return value, flag
        if type(value_type)==tuple:
            for i, val_type in enumerate(value_type):
                if val_type == '+float':
                    try: value[i] = float(value[i])
                    except: flag = False
                flag = flag and (value[i] > 0)
            return value, flag
        if type(value_type)==list:
            flag = (value in value_type)
            if (not flag) and type(value_type[0])==int:
                try: value = int(float(value))
                except: pass
                flag = (value in value_type)
            return value, flag
        if value_type == 'filename':
            try: value = str(value)
            except: flag = False
            flag = flag and ('.' in value)
            return value, flag
        else:
            print('incorrect value')
            return value, False
        
class App(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self) 
        self.f = False
        self.DepositionButton.clicked.connect(self.deposition)
        self.update_model_Button.clicked.connect(self.update_model)
        self.settings_input = Settings(settings)
        self.table_settings.setModel(self.settings_input)
        self.update_model()
        self.R_Slider.valueChanged.connect(self.set_R_slider)
        self.k_Slider.valueChanged.connect(self.set_k_slider)
        self.NR_Slider.valueChanged.connect(self.set_NR_slider)
        self.R_step = 1
        self.k_step = 0.01
        self.NR_step = 0.01
        self.R_Slider.setRange(*np.multiply(self.model.R_bounds, 1/self.R_step))
        self.k_Slider.setRange(*np.multiply(self.model.k_bounds, 1/self.k_step))
        self.NR_Slider.setRange(*np.multiply(self.model.NR_bounds, 1/self.NR_step))
        self.R_disp.editingFinished.connect(self.set_R_line)
        self.k_disp.editingFinished.connect(self.set_k_line)
        self.NR_disp.editingFinished.connect(self.set_NR_line)
        self.R_Slider.setValue(self.R/self.R_step)
        self.k_Slider.setValue(self.k/self.k_step)
        self.NR_Slider.setValue(self.NR/self.NR_step)
        self.set_R(20)
        self.set_k(1.5)
        self.set_NR(1)
        
    def set_R_line(self):
        self.R = float(self.R_disp.text())
        self.set_R()
        
    def set_R_slider(self):
        self.R = float(self.R_Slider.value())*self.R_step
        self.set_R()
        
    def set_R(self, value=None):
        if not value:
            value = self.R
        else:
            self.R = value
        self.lcd_R.display(self.R)
        self.R_disp.setText(('%.'+str(int(np.log10(1/self.R_step)))+'f') % self.R)
        self.R_Slider.setValue(self.R/self.R_step)
        
    def set_k_line(self):
        self.k = float(self.k_disp.text())
        self.set_k()
        
    def set_k_slider(self):
        self.k = float(self.k_Slider.value())*self.k_step
        self.set_k()
        
    def set_k(self, value=None):
        if not value:
            value = self.k
        else:
            self.k = value
        self.lcd_k.display(self.k)
        self.k_disp.setText(('%.'+str(int(np.log10(1/self.k_step)))+'f') % self.k)
        self.k_Slider.setValue(self.k/self.k_step)
        
    def set_NR_line(self):
        self.NR = float(self.NR_disp.text())
        self.set_NR()
        
    def set_NR_slider(self):
        self.NR = float(self.NR_Slider.value())*self.NR_step
        self.set_NR()
        
    def set_NR(self, value=None):
        if not value:
            value = self.NR
        else:
            self.NR = value
        self.lcd_NR.display(self.NR)
        self.NR_disp.setText(('%.'+str(int(np.log10(1/self.NR_step)))+'f') % self.NR)
        self.NR_Slider.setValue(self.NR/self.NR_step)
        
    def update_model(self):
        settings = self.settings_input.settigs()
        self.model = functions.Model(**settings)
        try: 
            self.mesh_plot_vl.canvas.figure.axes[0].cla()
            self.source_plot_vl.canvas.figure.axes[0].cla()
        except:
            fig1 = Figure()
            self.mesh_plot_vl.canvas = FigureCanvas(fig1)
            self.mesh_plot_vl.addWidget(self.mesh_plot_vl.canvas)
            toolbar1 = NavigationToolbar(self.mesh_plot_vl.canvas, self.mesh_plot)
            self.mesh_plot_vl.addWidget(toolbar1)
            self.mesh_plot_vl.canvas.figure.add_subplot(111)
            fig2 = Figure()
            self.source_plot_vl.canvas = FigureCanvas(fig2)
            self.source_plot_vl.addWidget(self.source_plot_vl.canvas)
            toolbar2 = NavigationToolbar(self.source_plot_vl.canvas, self.source_plot)
            self.source_plot_vl.addWidget(toolbar2)
            self.source_plot_vl.canvas.figure.add_subplot(111)
        ax1 = self.mesh_plot_vl.canvas.figure.axes[0]
        ax1.plot(self.model.substrate_rect_x, self.model.substrate_rect_y, color='black')
        ax1.plot(np.reshape(self.model.substrate_coords_map_x, (-1, 1)), 
                   np.reshape(self.model.substrate_coords_map_y, (-1, 1)), 'x', 
                   label='mesh point')
        ax1.set_title('Substrate')
        ax1.set_xlabel('x, mm')
        ax1.set_ylabel('y, mm')  
        self.mesh_plot_vl.canvas.draw()
        
        ax2 = self.source_plot_vl.canvas.figure.axes[0]
        ax2.contourf(self.model.deposition_coords_map_x, self.model.deposition_coords_map_y, 
                     self.model.deposition_coords_map_z, 100)

        ax2.plot(self.model.holder_circle_inner_x, self.model.holder_circle_inner_y, linewidth=2, 
         color='black', linestyle='--')

        ax2.plot(self.model.holder_circle_outer_x, self.model.holder_circle_outer_y, linewidth=2, 
         color='black')

        ax2.plot(self.model.deposition_rect_x, self.model.deposition_rect_y, linewidth=2, color='green')
        #plt.colorbar()
        #plt.xlim((min(deposition_rect_x), max(deposition_rect_x)))
        #plt.ylim((min(deposition_rect_y), max(deposition_rect_y)))
        ax2.set_xlabel('x, mm')
        ax2.set_ylabel('y, mm')
        self.source_plot_vl.canvas.draw()
        
    def deposition(self):
        self.update_model()
        I, heterogeneity, I_err = self.model.deposition(self.R, self.k, self.NR, 3)
        try: 
            self.film_vl.canvas.figure.axes[0].cla()
        except:
            fig = Figure()
            self.film_vl.canvas = FigureCanvas(fig)
            self.film_vl.addWidget(self.film_vl.canvas)
            toolbar = NavigationToolbar(self.film_vl.canvas, self.film_plot)
            self.film_vl.addWidget(toolbar)
            self.film_vl.canvas.figure.add_subplot(111)
        ax1f = self.film_vl.canvas.figure.axes[0]
        ax1f.contourf(self.model.substrate_coords_map_x, 
                      self.model.substrate_coords_map_y, I/I.max())
        #fig.clim(I.min()/I.max(), 1)
        #fig.colorbar(ax1f)
        ax1f.set_xlabel('x, mm')
        ax1f.set_ylabel('y, mm')
        ax1f.set_title(f'Film heterogeneity $H = {round(heterogeneity,2)}\\%$')
        self.film_vl.canvas.draw()
    