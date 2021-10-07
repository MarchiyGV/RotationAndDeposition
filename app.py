from math import ceil, floor
from PyQt5.QtCore import (Qt, QSortFilterProxyModel, pyqtSignal, pyqtSlot, 
QThread, QModelIndex, QAbstractTableModel, QVariant, QCoreApplication)
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QInputDialog,
    QFileDialog,
    QAbstractItemView,
    QErrorMessage,
    QMessageBox,
    QShortcut,
    QLineEdit,
    QTableView
)
from PyQt5.QtGui import QKeySequence, QFocusEvent, QKeyEvent, QCursor
import matplotlib
import matplotlib.ticker as ticker
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from numpy import (array, multiply, log10, mean, min)
import os
from multiprocessing import freeze_support

import exception_hooks
import design
import functions 
from global_parameters import *
from settings import *

def round_to_1(x):
   return round(x, -int(floor(log10(abs(x)))))

def waiting_effects(function):
    def new_function(self):
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        res = False
        try:
            return function(self)
        except Exception as e:
            self.error(f'{type(e).__name__}: {str(e)}')
        finally:
            QApplication.restoreOverrideCursor()
    return new_function

#pyuic5 "C:\Users\Георгий\Desktop\ФТИ\RotationAndDeposition\gui.ui" -o "C:\Users\Георгий\Desktop\ФТИ\RotationAndDeposition/design.py"
font = {'family' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)
matplotlib.use('QT5Agg')
    
class Thread(QThread):
    msg = pyqtSignal(str)
    def __init__(self, func_run, args_run, parent=None):
        super().__init__(parent)
        self.func_run = func_run
        self.args_run = args_run
        
    def run(self):
        self.func_run(*self.args_run)
        
class Opt(QThread):
    msg = pyqtSignal(str)
    def __init__(self, func_run, args_run, parent=None):
        super().__init__(parent)
        self.func_run = func_run
        self.args_run = args_run
        
    def run(self):
        self.func_run(*self.args_run)
        
class Dep_log(QAbstractTableModel):
    
    headers = ['R', 'k', 'N', 'omega', 'proc.\ntime', 'comp.\ntime', 'het.', 'err']
    
    def __init__(self, data=[], parent=None):
        super().__init__(parent)
        self.data = data
        self.d = [0, 2, 2, 2, 0, 1, 2, 2]
                  
    def set_accuracy(self, i, d):
        self.d[i] = d
        
    def headerData(self, section: int, orientation: Qt.Orientation, role: int):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.headers[section]
            else:
                return str(1+section)

    def columnCount(self, parent=None):
        return len(self.headers)

    def rowCount(self, parent=None):
        return len(self.data)
    
    def data(self, index: QModelIndex, role: int):
        if not index.isValid():
            return QVariant()
        if index.isValid():
            if role == Qt.DisplayRole:
                row = index.row()
                col = index.column()
                return str(self.data[row][col])
        
            
    def flags(self, index):
        return Qt.ItemIsEnabled
    
    def append(self, row):
        if len(row)==self.columnCount():
            for i in range(self.columnCount()):
                row[i] = round(row[i], int(self.d[i]))
            self.data.append(row)
        else:
            raise ValueError

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
        print('fwfwe')
        if event.key() == Qt.Key_Return:
            
            self.clearFocus()
        QLineEdit.keyPressEvent(self, event)
            

class App(QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self) 
        self.warnbox = QErrorMessage(self)
        self.errorbox = QMessageBox(self)
        self.deposition_log = Dep_log()
        self.dep_log_table.setModel(self.deposition_log)
        self.dep_log_table.resizeColumnsToContents()
        self.dep_log_table.horizontalHeader().setStretchLastSection(True)
        self.table_settings.setEditTriggers(QAbstractItemView.CurrentChanged)
        self.table_settings_opt.setEditTriggers(QAbstractItemView.CurrentChanged)
        self.DepositionButton.clicked.connect(self.deposition)
        self.cancel_dep_button.clicked.connect(self.deposition_stop)
        self.update_model_Button.clicked.connect(self.update_model)
        self.save_settings_Button.clicked.connect(self.save_settings)
        self.open_settings_Button.clicked.connect(self.open_settings)
        self.shortcut_deposite = QShortcut('Return', self.Deposition)
        self.shortcut_update = QShortcut('Return', self.Model)
        self.shortcut_deposite.activated.connect(self.DepositionButton.clicked.emit)
        self.shortcut_update.activated.connect(self.update_model_Button.clicked.emit)
        self.model = functions.Model()
        self.model.log_signal.connect(self.model_log)
        self.save_path = 'saves/'   
        success = self.update_settings(self.save_path+'settings.xlsx')
        if success:
            self.set_delegates(self.table_settings, self.model_settings)
            self.set_delegates(self.table_settings_opt, self.opt_settings)
        else: 
            self.disable_model(True)
        self.R_Slider.valueChanged.connect(self.plot_geometry_upd)
        self.R_Slider.valueChanged.connect(self.set_R)
        self.k_Slider.valueChanged.connect(self.set_k)
        self.NR_Slider.valueChanged.connect(self.set_NR)
        self.R_disp.valueChanged.connect(self.plot_geometry_upd)
        self.R_disp.valueChanged.connect(self.set_R)
        self.k_disp.valueChanged.connect(self.set_k)
        self.NR_disp.valueChanged.connect(self.set_NR)
        self.thick_edit.editingFinished.connect(self.set_h)
        if success:
                self.set_R(mean(self.model.R_bounds))
                self.set_k(mean(self.model.k_bounds))
                self.set_NR(min(self.model.NR_bounds))

        self.h = 100
        self.thick_edit.setText(str(self.h))
        self.optimiseButton.clicked.connect(self.optimisation)
        self.optimisationLog.setText('Log: \n')
        self.settings.upd_signal.connect(self.update_settings_dependansies)
        self.settings.editingFinished.connect(self.select) #fix one strange problem
        self.p_dep_bar.setValue(0)
        self.InputWidget.currentChanged.connect(self.tabChanged)
        self.meshBox.stateChanged.connect(self.plot_mesh)
        self.tolerance_edit.focused.connect(self.disable_return_shortcut)
        self.tolerance_edit.unfocused.connect(self.enable_return_shortcut)
        self.sub_res_edit.focused.connect(self.disable_return_shortcut)
        self.sub_res_edit.unfocused.connect(self.enable_return_shortcut)
        self.thick_edit.focused.connect(self.disable_return_shortcut)
        self.thick_edit.unfocused.connect(self.enable_return_shortcut)
        self.tolerance_edit.editingFinished.connect(self.set_settings)
        self.sub_res_edit.editingFinished.connect(self.set_settings)
        
    @pyqtSlot()    
    def disable_return_shortcut(self):
        self.shortcut_deposite.setEnabled(False)
        self.shortcut_update.setEnabled(False)
        
    @pyqtSlot()    
    def enable_return_shortcut(self):
        self.shortcut_deposite.setEnabled(True)
        self.shortcut_update.setEnabled(True)
        
    @pyqtSlot()    
    def set_settings(self):
        sender = self.sender()
        d = {'tolerance_edit': 'point_tolerance', 
             'sub_res_edit':  'substrate_res'}
        varname = d[sender.objectName()]
        value = sender.text()
        success = self.settings.setNamedData(varname, value)
        if success:
            self.update_model()
    def resizeEvent(self, event):
        QMainWindow.resizeEvent(self, event)
        index = self.InputWidget.currentIndex()
        if index == 0:
            self.mesh_plot_vl.canvas.figure.tight_layout()
            self.mesh_plot_vl.canvas.figure.tight_layout()
        if index == 1:
            self.geometry_vl.canvas.figure.tight_layout()
            self.geometry_vl.canvas.figure.tight_layout()
    
    @pyqtSlot(int)    
    def tabChanged(self, index):
        if index == 0:
            self.mesh_plot_vl.canvas.figure.tight_layout()
            self.mesh_plot_vl.canvas.figure.tight_layout()
        if index == 1:
            self.geometry_vl.canvas.figure.tight_layout()
            self.geometry_vl.canvas.figure.tight_layout()
        
    @pyqtSlot(int)
    def select(self, row): #fix one strange problem
        selection = self.table_settings.model().index(row, self.settings.index_name)
        self.table_settings.setCurrentIndex(selection)
            
    def disable_model(self, flag):
        self.DepositionButton.setDisabled(flag)
        self.optimiseButton.setDisabled(flag)
        self.shortcut_deposite.setEnabled(self.InputWidget.currentIndex()==1 and not flag)
        
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
                
    @pyqtSlot(int)
    def update_settings_dependansies(self, row=0):
        for i in range(self.settings.rowCount()):
            self.table_settings.setRowHidden(i, (not self.settings.isVisible(i)))
        for i in range(self.settings.rowCount()):
            self.table_settings_opt.setRowHidden(i, (not self.settings.isVisible(i)))
        
    def update_settings(self, fname):
        try:
            self.settings = Settings.open_file(fname)
        except:
            self.error('Неверный формат файла с настройками')
            return False
        success = self.update_model()
        if not success:
            return False
        ### model tab
        self.model_settings = QSortFilterProxyModel()
        self.model_settings.setSourceModel(self.settings)
        self.model_settings.setFilterKeyColumn(self.settings.index_group)
        self.model_settings.setFilterRegExp('(model)|(sys)|(numerical)')
        self.table_settings.setModel(self.model_settings)
        for i in range(self.settings.columnCount()):
            self.table_settings.setColumnHidden(i, (not i in self.settings.indexes_visible))
        self.table_settings.verticalHeader().setVisible(False)
        self.table_settings.resizeColumnsToContents()
        i = self.settings.index_value
        resize = 2
        self.table_settings.setColumnWidth(i, self.table_settings.columnWidth(i)*resize)
        i = self.settings.index_units
        resize = 0.5
        self.table_settings.setColumnWidth(i, self.table_settings.columnWidth(i)*resize)
        self.table_settings.horizontalHeader().setStretchLastSection(True)
        ### optimize tab
        self.opt_settings = QSortFilterProxyModel()
        self.opt_settings.setSourceModel(self.settings) 
        self.opt_settings.setFilterKeyColumn(self.settings.index_group)
        self.opt_settings.setFilterRegExp('(minimisation)|(sys)|(numerical)')
        self.table_settings_opt.setModel(self.opt_settings)
        for i in range(self.settings.columnCount()):
            self.table_settings_opt.setColumnHidden(i, (not i in self.settings.indexes_visible))
        self.table_settings_opt.verticalHeader().setVisible(False)
        self.table_settings_opt.resizeColumnsToContents()
        ### 
        self.update_settings_dependansies()
        self.settings.upd_signal.connect(self.update_settings_dependansies)
        self.settings.upd_signal.connect(self.select) #fix one strange problem
        self.settings.editing.connect(self.disable_return_shortcut)
        self.settings.editingFinished.connect(self.enable_return_shortcut)
        return True
        
    @pyqtSlot()
    def save_settings(self):
        name, flag = QInputDialog.getText(self, 'Input Dialog',
            'File name:')
        if flag:
            self.update_model()
            self.settings.save(self.save_path+name)
            
    @pyqtSlot()    
    def open_settings(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', os.getcwd()+'/'+self.save_path+'settings.xlsx')[0]
        if fname:
            self.update_settings(fname)
        else:
            pass
        
    def update_sliders(self):
        range_R = multiply(self.model.R_bounds, 1/self.model.R_step).astype(int)
        range_k = multiply(self.model.k_bounds, 1/self.model.k_step).astype(int)
        range_NR = multiply(self.model.NR_bounds, 1/self.model.NR_step).astype(int)
        self.R_Slider.setRange(*range_R)
        self.R_disp.setRange(*self.model.R_bounds)
        self.k_Slider.setRange(*range_k)
        self.k_disp.setRange(*self.model.k_bounds)
        self.NR_Slider.setRange(*range_NR)
        self.NR_disp.setRange(*self.model.NR_bounds)
        self.R_disp.setDecimals(int(ceil(log10(1/self.model.R_step))))
        self.R_disp.setSingleStep(self.model.R_step)
        self.k_disp.setSingleStep(self.model.k_step)
        self.NR_disp.setSingleStep(self.model.NR_step)
        self.k_disp.setDecimals(int(ceil(log10(1/self.model.k_step))))
        self.NR_disp.setDecimals(int(ceil(log10(1/self.model.NR_step))))
  
    @pyqtSlot()
    def set_R(self, value=None):
        sender = self.sender()
        if not sender is None:
            if sender == self.R_Slider:
                self.R = float(self.R_Slider.value())*self.model.R_step
                self.R_disp.blockSignals(True)
                self.R_disp.setValue(self.R)
                self.R_disp.blockSignals(False)
            elif sender == self.R_disp:
                self.R = float(self.R_disp.value())
                self.R_Slider.blockSignals(True)
                self.R_Slider.setValue(int(self.R/self.model.R_step))
                self.R_Slider.blockSignals(False)
        else:
            self.R_disp.blockSignals(True)
            self.R_Slider.blockSignals(True)
            self.R = value
            self.R_Slider.setValue(int(self.R/self.model.R_step))
            self.R_disp.setValue(self.R)
            self.R_disp.blockSignals(False)
            self.R_Slider.blockSignals(False)
            
    @pyqtSlot()        
    def set_k(self, value=None):
        sender = self.sender()
        if not sender is None:
            if sender == self.k_Slider:
                self.k = float(self.k_Slider.value())*self.model.k_step
                self.k_disp.blockSignals(True)
                self.k_disp.setValue(self.k)
                self.k_disp.blockSignals(False)
            elif sender == self.k_disp:
                self.k = float(self.k_disp.value())
                self.k_Slider.blockSignals(True)
                self.k_Slider.setValue(int(self.k/self.model.k_step))
                self.k_Slider.blockSignals(False)
        else:
            self.k_disp.blockSignals(True)
            self.k_Slider.blockSignals(True)
            self.k = value
            self.k_disp.setValue(self.k)
            self.k_Slider.setValue(int(self.k/self.model.k_step))
            self.k_disp.blockSignals(False)
            self.k_Slider.blockSignals(False)
            
    @pyqtSlot() 
    def set_NR(self, value=None):
        sender = self.sender()
        if not sender is None:
            if sender == self.NR_Slider:
                self.NR = float(self.NR_Slider.value())*self.model.NR_step
                self.NR_disp.blockSignals(True)
                self.NR_disp.setValue(self.NR)
                self.NR_disp.blockSignals(False)
            elif sender == self.NR_disp:
                self.NR = float(self.NR_disp.value())
                self.NR_Slider.blockSignals(True)
                self.NR_Slider.setValue(int(self.NR/self.model.NR_step))
                self.NR_Slider.blockSignals(False)
        else:
            self.NR_disp.blockSignals(True)
            self.NR_Slider.blockSignals(True)
            self.NR = value
            self.NR_disp.setValue(self.NR)
            self.NR_Slider.setValue(int(self.NR/self.model.NR_step))
            self.NR_disp.blockSignals(False)
            self.NR_Slider.blockSignals(False)
    
    @pyqtSlot()    
    def set_h(self):
        self.h = float(self.sender().text())
        self.thick_edit.setText(str(self.h))
        
    @pyqtSlot(str, str) 
    def model_log(self, msg, type):
        self.model_info.append(msg)
        
    @pyqtSlot(str, str) 
    def warn(self, msg, type=None):
        self.warnbox.showMessage(msg, type)
        self.warnbox.exec_()
        
    def error(self, msg):
        self.errorbox.setText(msg)
        self.errorbox.exec_()
    
    @pyqtSlot()  
    @waiting_effects 
    def update_model(self):
        self.model_info.setText('')
        settings = self.settings.wrap()
        self.model.update(**settings)
        self.model.deposition.finished.connect(self.deposition_plot)
        self.model.deposition.progress_signal.connect(self.deposition_progress)
        self.model.deposition.msg_signal.connect(self.deposition_logout)
        if self.model.success:
            self.disable_model(False)
        else:
            self.error('Ошибка при инициализации модели')
            self.disable_model(True)
            return False
        if self.model.rotation_type == 'Solar':
            self.set_k(1)
            self.k_Slider.setDisabled(True)
            self.k_disp.setDisabled(True)
        else:
            self.k_Slider.setDisabled(False)
            self.k_disp.setDisabled(False)
        self.optimizer = functions.Optimizer(self.model.deposition)
        self.disable_model(False)
        self.update_sliders()
        self.plot_model()
        self.plot_geometry()
        step = [self.model.R_step, self.model.k_step, self.model.NR_step]
        for i in range(len(step)):
            d = ceil(log10(1/step[i]))
            self.deposition_log.set_accuracy(i, d)
        self.tolerance_edit.setText(str(self.model.point_tolerance*100))
        self.sub_res_edit.setText(str(self.model.substrate_res))
        self.plot_mesh()
        return True
    
    def plot_model(self):
        try: 
            self.mesh_plot_vl.canvas.figure.axes[0].cla()
            self.mesh_plot_vl.canvas.figure.axes[1].cla()
        except:
            chosen_value = 10
            fig = Figure()
            self.mesh_plot_vl.canvas = FigureCanvas(fig)
            self.mesh_plot_vl.addWidget(self.mesh_plot_vl.canvas)
            toolbar = NavigationToolbar(self.mesh_plot_vl.canvas, self.mesh_plot)
            self.mesh_plot_vl.addWidget(toolbar)
            self.mesh_plot_vl.canvas.figure.add_subplot(121)
            self.mesh_plot_vl.canvas.figure.add_subplot(122)
        ax1 = self.mesh_plot_vl.canvas.figure.axes[0]
        ax1.plot(self.model.substrate_rect_x, self.model.substrate_rect_y, color='black')
        ax1.plot(self.model.xs, self.model.ys, 'x', label='mesh point')
        ax1.set_xlabel('x, mm')
        ax1.set_ylabel('y, mm')  
        ax1.set_aspect('equal')
        ax1.set_title('Сетка подложки\n')
        ax2 = self.mesh_plot_vl.canvas.figure.axes[1]
        im = ax2.contourf(self.model.deposition_coords_map_x, 
                     self.model.deposition_coords_map_y, 
                     self.model.deposition_coords_map_z, 100)

        ax2.plot(self.model.holder_circle_inner_x, 
                 self.model.holder_circle_inner_y, 
                 linewidth=2, color='black', linestyle='--')

        ax2.plot(self.model.holder_circle_outer_x, 
                 self.model.holder_circle_outer_y, 
                 linewidth=2, color='black')

        ax2.plot(self.model.deposition_rect_x, 
                 self.model.deposition_rect_y, 
                 linewidth=2, color='green')
        
        substrate_rect_x = array(self.model.substrate_rect_x)+mean(self.model.R_bounds)
        substrate_rect_y = array(self.model.substrate_rect_y)
        ax2.plot(substrate_rect_x, substrate_rect_y, color='black')
        ax2.text(substrate_rect_x.mean(), substrate_rect_y.max()*1.1,
                 'substrate', size=12, ha='center')
        ax2.arrow(0, 0, mean(self.model.R_bounds), 0, 
                  width=0.5, head_length = 4, head_width=4, color='black')
        ax2.text(mean(self.model.R_bounds)-5, 5, 'R', size=12, ha='right')
        try:
            self.model_cbar.remove()
        except AttributeError:
            pass
        
        self.model_cbar = self.mesh_plot_vl.canvas.figure.colorbar(im, fraction=0.046, 
                                                                   pad=0.04)
        self.model_cbar.set_label('nm/min')
        ax2.set_xlabel('x, mm')
        ax2.set_ylabel('y, mm')
        ax2.set_aspect('equal')
        ax2.set_title('Профиль источника\n')
        self.mesh_plot_vl.canvas.figure.tight_layout()
        self.mesh_plot_vl.canvas.draw()
        
    @pyqtSlot() 
    def plot_geometry_upd(self):
        ax = self.geometry_vl.canvas.figure.axes[0]
        substrate_rect_x = array(self.model.substrate_rect_x)+self.R
        #substrate_rect_y = array(self.model.substrate_rect_y)
        self.geometry_rect_ref.set_xdata(substrate_rect_x)
        #self.geometry_plot_ref.set_ydata(substrate_rect_y)
        self.geometry_text_ref.set_x(substrate_rect_x.mean())
        #self.geometry_text_ref.set_y(substrate_rect_y.max()*1.1)
        self.geometry_arrow_ref.remove()
        self.geometry_arrow_ref = ax.arrow(0, 0, self.R, 0, 
                  width=0.5, head_length = 4, head_width=4, color='black')
        self.geometry_R_ref.set_x(self.R-5)
        self.geometry_vl.canvas.draw()

    @pyqtSlot() 
    def plot_geometry(self):
        try:
            self.geometry_vl.canvas.figure.clf()
        except:
            fig = Figure()
            self.geometry_vl.canvas = FigureCanvas(fig)
            self.geometry_vl.addWidget(self.geometry_vl.canvas)
            self.toolbar2 = NavigationToolbar(self.geometry_vl.canvas, self.geometry_plot)
            self.geometry_vl.addWidget(self.toolbar2)
        self.geometry_vl.canvas.figure.add_subplot(121)
        self.geometry_vl.canvas.figure.add_subplot(122)
        ax2f = self.geometry_vl.canvas.figure.axes[0]
        im = ax2f.contourf(self.model.deposition_coords_map_x, self.model.deposition_coords_map_y, 
                         self.model.deposition_coords_map_z, 100)
    
        ax2f.plot(self.model.holder_circle_inner_x, self.model.holder_circle_inner_y, linewidth=2, 
             color='black', linestyle='--')
    
        ax2f.plot(self.model.holder_circle_outer_x, self.model.holder_circle_outer_y, linewidth=2, 
             color='black')
    
        ax2f.plot(self.model.deposition_rect_x, self.model.deposition_rect_y, linewidth=2, color='green')
        try:
            R = self.R
        except AttributeError:
            R = mean(self.model.R_bounds)
        substrate_rect_x = array(self.model.substrate_rect_x)+R
        substrate_rect_y = array(self.model.substrate_rect_y)
        rect_refs = ax2f.plot(substrate_rect_x, substrate_rect_y, color='black')
        
        text_refs = ax2f.text(substrate_rect_x.mean(), substrate_rect_y.max()*1.1,
                 'substrate', size=12, ha='center')
        arrow_refs = ax2f.arrow(0, 0, R, 0, 
                  width=0.5, head_length = 4, head_width=4, color='black')
        R_refs = ax2f.text(R-5, 5, 'R', size=12, ha='right')
        self.geometry_rect_ref = rect_refs[0]
        self.geometry_text_ref = text_refs
        self.geometry_arrow_ref = arrow_refs
        self.geometry_R_ref = R_refs
        ax2f.set_xlabel('x, mm')
        ax2f.set_ylabel('y, mm')
        ax2f.set_aspect('equal')
        ax2f.set_title('Геометрия источника\n')
        ax1f = self.geometry_vl.canvas.figure.axes[1]
        ax1f.set_xlabel('x, mm')
        ax1f.set_ylabel('y, mm')
        ax1f.set_title('Толщина плёнки\n')
        ax1f.set_aspect('equal')
        try:
            self.geometry_cbar.remove()
        except AttributeError:
            pass
        except ValueError:
            pass
        self.geometry_cbar = self.geometry_vl.canvas.figure.colorbar(im,
                                                                     fraction=0.046, 
                                                                     pad=0.04,
                                                                     ax=ax2f)
        self.geometry_cbar.set_label('nm/min')
        self.geometry_vl.canvas.figure.tight_layout()
        self.geometry_vl.canvas.draw()
    
    @pyqtSlot()
    def deposition(self):
        self.deposition_output.setText('')
        self.dep_terminated=False
        self.dep_msg = []
        self.DepositionButton.setDisabled(True)
        self.cancel_dep_button.setDisabled(False)
        args = [self.R, self.k, self.NR, 1, self.model.alpha0_sub, 
                self.model.point_tolerance, self.model.cores]
        self.model.deposition.task(*args)
        self.p_dep_bar.setValue(0)
        self.model.deposition.start()
        
    @pyqtSlot()
    def deposition_stop(self):
        self.dep_terminated=True
        self.model.deposition.terminate()  
               
    @pyqtSlot(float)
    def deposition_progress(self, progress):
        self.p_dep_bar.setValue(int(round(progress*100)))
        
    @pyqtSlot(str)
    def deposition_logout(self, s):
        if s not in self.dep_msg:
            self.dep_msg.append(s)
            s = s.replace('\n', '').replace('  ', ' ').replace('  ', ' ')
            self.deposition_output.append(s+'\n')
        
    @pyqtSlot()    
    def deposition_plot(self):
        self.DepositionButton.setDisabled(False)
        self.cancel_dep_button.setDisabled(True)
        if self.dep_terminated:
            return False
        I = self.model.deposition.hs
        if I is None:
            return False
        ###output
        het = self.model.heterogeneity(I)
        thickness = I.mean()
        omega = thickness/self.h
        proc_time = self.NR/omega
        t = self.model.deposition.time[-1]
        self.deposition_output.append(f'Неоднородность: ({round(het,2)} +- {round_to_1(4*self.model.point_tolerance*100)})%')
        if het < self.model.point_tolerance*100*4/10:
            msg = f'\nРезултат расчёта неоднородности может быть неверным, так как полученная неоднородность {round(het,2)}% меньше заданной точности расчёта {4*self.model.point_tolerance*100}% более чем на порядок. \nНеобходимо уменьшить параметр "Точность в точке"' 
            self.warn(msg)
            self.deposition_output.append(msg)
        elif het < self.model.point_tolerance*4*100:
            self.deposition_output.append(f'\nДостоверность расчёта неоднородности не может быть гарантированна, так как полученная неоднородность {round(het,2)}% меньше максимально возможной абсолютной погрешности +-{4*self.model.point_tolerance*100}%, однако обычно эта оценка погрешности превосходит реальную погрешность на порядок (т.е. оптимистичный прогноз абсолютной погрешности +-{round_to_1(self.model.point_tolerance*100*4/10)}%). Для достоверности рекомендуется уменьшить параметр "Точность в точке"')
        n = int(ceil(log10(1/omega)))
        
        
        self.deposition_log.append([self.R, self.k, self.NR, omega, proc_time, t, het, self.model.point_tolerance*4*100])
        self.dep_log_table.model().layoutChanged.emit()
        for i in range(self.deposition_log.columnCount()-1):
            self.dep_log_table.resizeColumnToContents(i)
        self.dep_log_table.horizontalHeader().setStretchLastSection(True)
        
        if n <= omega_decimals:
            self.deposition_output.append(f'\nУгловая скорость: %.{omega_decimals}f оборотов/мин.' % omega)
        else:
            self.deposition_output.append('\nУгловая скорость: %.2fe-%d оборотов/мин.' % (omega*(10**n), n))
        self.deposition_output.append('\nВремя процесса: %d мин.' % (proc_time))
        self.deposition_output.append(f'\nВремя расчёта: {round(t,2)}s')
        if omega>self.model.omega_s_max:
            self.deposition_output.append('\n!!! Превышена максимальная угловая скорость солнца')
        if omega*self.k>self.model.omega_p_max:
            self.deposition_output.append('\n!!! Превышена максимальная угловая скорость планеты')
        ### plotting
        ax1f = self.geometry_vl.canvas.figure.axes[1]
        im = ax1f.tricontourf(self.model.xs, self.model.ys, I/I.max())
        ax1f.plot(self.model.substrate_rect_x, self.model.substrate_rect_y, 
                  color='black', linewidth=7)
        if self.meshBox.isChecked():
            self.mesh.remove()
            self.mesh = ax1f.plot(self.model.xs, self.model.ys, '.', color=mesh_color)[0]
        
        @ticker.FuncFormatter
        def major_formatter(x, pos):
            z = x*100
            if het > 5:
                return "%d" % z
            elif het > 1:
                return "%.1f" % z
            else:
                return "%.2f" % z
        try:
            self.film_cbar.remove()
        except AttributeError:
            pass
        except ValueError:
            pass
        self.film_cbar = self.geometry_vl.canvas.figure.colorbar(im,
                                                                 fraction=0.046, 
                                                                 pad=0.04,
                                                                 ax=ax1f,
                                                                 format=major_formatter)
        self.film_cbar.set_label('% $h_{max}$')
        self.geometry_vl.canvas.figure.tight_layout()
        self.geometry_vl.canvas.draw()
        
    @pyqtSlot(int)    
    def plot_mesh(self, state=None):
        if state is None:
            state = self.meshBox.checkState()
            if state == 0:
                return
        if state==2:
            ax = self.geometry_vl.canvas.figure.axes[1]
            self.mesh = ax.plot(self.model.xs, self.model.ys, '.', color=mesh_color)[0]
            self.geometry_vl.canvas.draw()
        elif state==0:
            self.mesh.remove()
            self.geometry_vl.canvas.draw()
            
    @pyqtSlot()    
    def optimisation(self): 
        self.optimiseButton.setDisabled(True)
        self.cancelOptimiseButton.setDisabled(False)      
        self.optimisationLog.setText('Log: \n') 
        args = [self.model.heterogeneity, self.model.alpha0_sub, 
                self.model.point_tolerance, self.model.cores, 
                self.model.R_bounds, self.model.k_bounds, self.model.NR_bounds, 
                self.model.R_min_step, self.model.k_min_step, 
                self.model.NR_min_step, 
                self.model.R_step, self.model.k_step, self.model.NR_step,
                self.model.R_mc_interval, self.model.k_mc_interval, 
                self.model.NR_mc_interval, self.model.x0, self.model.minimizer, 
                self.model.mc_iter, self.model.T, self.model.verbose]
        self.optimisation_thread = Thread(self.optimizer.optimisation, args)
        self.cancelOptimiseButton.clicked.connect(self.optimisation_stop)
        self.optimisation_thread.finished.connect(self.optimisation_output)
        self.optimisation_thread.msg.connect(self.optimisation_log)
        self.optimizer.upd_signal.connect(self.optimisation_log)
        self.optimisation_thread.start()

    @pyqtSlot(str)
    def optimisation_log(self, message):
        self.optimisationLog.append(message)
        
    @pyqtSlot()
    def optimisation_stop(self):
        self.optimisation_thread.terminate()
        
    @pyqtSlot()    
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
    freeze_support()
    import sys
    main()
