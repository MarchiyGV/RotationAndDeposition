from PyQt5.QtCore import Qt, QSortFilterProxyModel, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QInputDialog,
    QFileDialog,
    QAbstractItemView
)

import matplotlib
matplotlib.use('QT5Agg')
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from numpy import (array, multiply, log10, reshape)
import os
import time
import multiprocessing as mp

import design
import functions 
from settings import *

#pyuic5 "C:\Users\Георгий\Desktop\ФТИ\RotationAndDeposition\gui.ui" -o "C:\Users\Георгий\Desktop\ФТИ\RotationAndDeposition/design.py"

class Task:
    def __init__(self, func_run, args_run, func_post, args_post, priority=QThread.InheritPriority):
        self.func_post = func_post
        self.args_post = args_post
        self.thread = Thread(func_run, args_run)
        self.thread.finished.connect(self.post)
        self.priority = priority
    
    def __call__(self):
        self.thread.start(self.priority)
    
    def kill(self):
        time.sleep(0.3)
        self.thread.terminate()
        self.thread.exit()
        
    def post(self):
        self.func_post(*self.args_post)
        
    
class Thread(QThread):
    def __init__(self, func_run, args_run):
        super().__init__()
        self.func_run = func_run
        self.args_run = args_run
        
    def run(self):
        self.func_run(*self.args_run)
        

class App(QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self) 
        self.deposition_task = Task(self.deposition, [], 
                                    self.deposition_plot, [], 
                                    QThread.TimeCriticalPriority)
        self.table_settings.setEditTriggers(QAbstractItemView.CurrentChanged)
        self.table_settings_opt.setEditTriggers(QAbstractItemView.CurrentChanged)
        self.DepositionButton.clicked.connect(self.deposition_task)
        self.update_model_Button.clicked.connect(self.update_model)
        self.save_settings_Button.clicked.connect(self.save_settings)
        self.open_settings_Button.clicked.connect(self.open_settings)
        self.save_path = 'saves/'       
        self.update_settings(self.save_path+'settings.xlsx')
        self.set_delegates(self.table_settings, self.model_settings)
        self.set_delegates(self.table_settings_opt, self.opt_settings)
        self.R_Slider.valueChanged.connect(self.set_R_slider)
        self.k_Slider.valueChanged.connect(self.set_k_slider)
        self.NR_Slider.valueChanged.connect(self.set_NR_slider)
        self.update_model()
        self.R_disp.editingFinished.connect(self.set_R_line)
        self.k_disp.editingFinished.connect(self.set_k_line)
        self.NR_disp.editingFinished.connect(self.set_NR_line)
        self.thick_edit.editingFinished.connect(self.set_h)
        self.set_R(25)
        self.set_k(1.5)
        self.set_NR(1)
        self.h = 100
        self.thick_edit.setText(str(self.h))
        self.optimiseButton.clicked.connect(self.optimisation_start)
        self.optimisationLog.setText('Log: \n')
        self.DepositionButton.setText('&Напыление')
        
    def set_delegates(self, table_view, proxy_model):
        l = 0
        while proxy_model.index(l,self.settings.index_type).isValid():
           l+=1
        for i in range(l):
            type_ = proxy_model.data(proxy_model.index(i,self.settings.index_type), Qt.DisplayRole)
            if 'cases' in type_:
                d = {}
                exec(type_, d) # cases = [..., ..., ...]
                items = d['cases']
                for j, item in enumerate(items):
                    items[j] = str(item)
                if 'labels' in d.keys():
                    labels = d['labels']
                    table_view.setItemDelegateForRow(i, DropboxDelegate(table_view, items, labels))
                else:
                    table_view.setItemDelegateForRow(i, DropboxDelegate(table_view, items))
            if type_ == 'bool':
                table_view.setItemDelegateForRow(i, YesNoDelegate(table_view))
            if type_ == 'filename':
                table_view.setItemDelegateForRow(i, OpenFileDelegate(table_view))
        
    def update_settings(self, fname):
        self.settings = Settings.open_file(fname)
        ### model tab
        self.model_settings = QSortFilterProxyModel()
        self.model_settings.setSourceModel(self.settings)
        self.model_settings.setFilterKeyColumn(self.settings.index_group)
        self.model_settings.setFilterRegExp('(model)|(sys)|(numerical)')
        self.model_settings.sort(self.settings.index_group, Qt.DescendingOrder)
        self.table_settings.setModel(self.model_settings)
        for i in range(self.settings.columnCount()):
            self.table_settings.setColumnHidden(i, (not i in self.settings.indexes_visible))
        self.table_settings.verticalHeader().setVisible(False)
        self.table_settings.resizeColumnsToContents()
        ### optimize tab
        self.opt_settings = QSortFilterProxyModel()
        self.opt_settings.setSourceModel(self.settings)
        self.opt_settings.setFilterKeyColumn(self.settings.index_group)
        self.opt_settings.setFilterRegExp('(minimisation)|(sys)|(numerical)')
        self.opt_settings.sort(self.settings.index_group, Qt.DescendingOrder)
        self.table_settings_opt.setModel(self.opt_settings)
        for i in range(self.settings.columnCount()):
            self.table_settings_opt.setColumnHidden(i, (not i in self.settings.indexes_visible))
        self.table_settings_opt.verticalHeader().setVisible(False)
        self.table_settings_opt.resizeColumnsToContents()
        
        
    def save_settings(self):
        name, flag = QInputDialog.getText(self, 'Input Dialog',
            'File name:')
        if flag:
            self.update_model()
            self.settings.save(self.save_path+name)
        
    def open_settings(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', os.getcwd()+'/'+self.save_path+'settings.xlsx')[0]
        if fname:
            self.update_settings(fname)
            self.update_model()
        else:
            pass
        
    def update_sliders(self):
        self.R_Slider.setRange(*multiply(self.model.R_bounds, 1/self.model.R_step).astype(int))
        self.k_Slider.setRange(*multiply(self.model.k_bounds, 1/self.model.k_step).astype(int))
        self.NR_Slider.setRange(*multiply(self.model.NR_bounds, 1/self.model.NR_step).astype(int))
        self.R_disp.setDecimals(int(log10(1/self.model.R_step)))
        self.R_disp.setSingleStep(self.model.R_step)
        self.k_disp.setSingleStep(self.model.k_step)
        self.NR_disp.setSingleStep(self.model.NR_step)
        self.k_disp.setDecimals(int(log10(1/self.model.k_step)))
        self.NR_disp.setDecimals(int(log10(1/self.model.NR_step)))
        
    def set_R_line(self):
        self.R = float(self.R_disp.text())
        self.set_R()
        
    def set_R_slider(self):
        self.R = float(self.R_Slider.value())*self.model.R_step
        self.set_R()
        
    def set_R(self, value=None):
        if not value:
            value = self.R
        else:
            self.R = value
        self.R_disp.setValue(self.R)
        self.R_Slider.setValue(int(self.R/self.model.R_step))
        
    def set_k_line(self):
        self.k = float(self.k_disp.text())
        self.set_k()
        
    def set_k_slider(self):
        self.k = float(self.k_Slider.value())*self.model.k_step
        self.set_k()
        
    def set_k(self, value=None):
        if not value:
            value = self.k
        else:
            self.k = value
        self.k_disp.setValue(self.k)
        self.k_Slider.setValue(int(self.k/self.model.k_step))
        
    def set_NR_line(self):
        self.NR = float(self.NR_disp.text())
        self.set_NR()
        
    def set_NR_slider(self):
        self.NR = float(self.NR_Slider.value())*self.model.NR_step
        self.set_NR()
        
    def set_NR(self, value=None):
        if not value:
            value = self.NR
        else:
            self.NR = value
        self.NR_disp.setValue(self.NR)
        self.NR_Slider.setValue(int(self.NR/self.model.NR_step))
        
    def set_h(self):
        self.h = float(self.sender().text())
        self.thick_edit.setText(str(self.h))
        
    def update_model(self):
        settings = self.settings.wrap()
        self.model = functions.Model(**settings)
        self.update_sliders()
        try: 
            self.mesh_plot_vl.canvas.figure.axes[0].cla()
            self.source_plot_vl.canvas.figure.axes[0].cla()
        except:
            fig1 = Figure()
            self.mesh_plot_vl.canvas = FigureCanvas(fig1)
            self.mesh_plot_vl.addWidget(self.mesh_plot_vl.canvas)
            toolbar1 = NavigationToolbar(self.mesh_plot_vl.canvas, self.mesh_plot)
            #self.mesh_plot_vl.addWidget(toolbar1)
            self.mesh_plot_vl.canvas.figure.add_subplot(111)
            fig2 = Figure()
            self.source_plot_vl.canvas = FigureCanvas(fig2)
            self.source_plot_vl.addWidget(self.source_plot_vl.canvas)
            toolbar2 = NavigationToolbar(self.source_plot_vl.canvas, self.source_plot)
            #self.source_plot_vl.addWidget(toolbar2)
            self.source_plot_vl.canvas.figure.add_subplot(111)
        ax1 = self.mesh_plot_vl.canvas.figure.axes[0]
        ax1.plot(self.model.substrate_rect_x, self.model.substrate_rect_y, color='black')
        ax1.plot(reshape(self.model.substrate_coords_map_x, (-1, 1)), 
                   reshape(self.model.substrate_coords_map_y, (-1, 1)), 'x', 
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
        self.DepositionButton.setDisabled(True)
        self.I = self.model.deposition(self.R, self.k, self.NR, 1)
        
    def deposition_plot(self):
        I = self.I
        heterogeneity = self.model.heterogeneity(I)
        thickness = I.mean()
        omega = thickness/self.h
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
        ax1f.set_title(f'$\omega = {round(omega,1)}$ 1/min, film heterogeneity $H = {round(heterogeneity,2)}\\%$')
        self.film_vl.canvas.draw()
        self.DepositionButton.setDisabled(False)
        
    def optimisation_start(self):
        self.optimisation_task = Task(self.optimisation, [], 
                                    self.optimisation_output, [], 
                                    QThread.TimeCriticalPriority)
        
        self.cancelOptimiseButton.clicked.connect(self.optimisation_task.kill)
        self.optimisationLog.setText('Log: \n')    
        self.optimiseButton.setDisabled(True)
        self.cancelOptimiseButton.setDisabled(False)
        self.update_model()
        self.optimisation_task()
        
    def optimisation(self):
        opt = functions.Optimizer(self.model)
        opt.upd_signal.connect(self.update_log)
        opt.optimisation()
    
    def update_log(self, message):
        self.optimisationLog.append(message)
        
    def optimisation_output(self):
        self.optimiseButton.setDisabled(False)
        self.cancelOptimiseButton.setDisabled(True)
        
def main():
    print('app = main')
    app = QApplication(sys.argv)  
    window = App() 
    window.show()  # Показываем окно
    sys.exit(app.exec_())
        
if __name__ == '__main__': 
    print(mp.get_context())
    mp.freeze_support()# Если мы запускаем файл напрямую, а не импортируем
    import sys
    main()  # то запускаем функцию main()