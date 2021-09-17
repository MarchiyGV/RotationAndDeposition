from PyQt5.QtCore import QModelIndex, Qt, QSortFilterProxyModel, QAbstractTableModel, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QStyledItemDelegate,
    QMainWindow,
    QInputDialog,
    QFileDialog
)

import matplotlib
matplotlib.use('QT5Agg')
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from numpy import (array, multiply, log10, reshape)
from pandas import (DataFrame, read_excel)
import re
import os
import design
import functions 

#pyuic5 "C:\Users\Георгий\Desktop\ФТИ\RotationAndDeposition\gui.ui" -o "C:\Users\Георгий\Desktop\ФТИ\RotationAndDeposition/design.py"


class Settings(QAbstractTableModel):
    def __init__(self, data=[], parent=None):
        super().__init__(parent)
        self.data = data
        self.index_name = 0
        self.index_variableName = 1
        self.index_value = 2
        self.index_type = 3
        self.index_group = 4
        self.index_comment = 5 
        self.indexes_visible = [self.index_name, self.index_value]
        self.headers = ['Параметр', 'Переменная', 'Значение', 'Тип', 'Группа', 'Комментарий']
         

    def save(self, filename):
        df = DataFrame(self.data)
        df.to_excel(filename+'.xlsx')
        
    def headerData(self, section: int, orientation: Qt.Orientation, role: int):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.headers[section]
            else:
                return str(1+section)

    def columnCount(self, parent=None):
        return len(self.data[0])

    def rowCount(self, parent=None):
        return len(self.data)
    
    def data(self, index: QModelIndex, role: int):
        if role == Qt.ToolTipRole:
            row=index.row()
            return self.data[row][self.index_comment]
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
        if index.column()==self.index_name:
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
            flag = flag and (value >= 0) and (value <= 1)
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
            '''
            d = {}
            exec(value_type, d) # cases = [..., ..., ...]
            value_type = d['cases']
            flag = (value in value_type)
            '''
            print(value)
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
    
class DropboxDelegate(QStyledItemDelegate):
    def __init__(self, wiget, items):
        super().__init__(wiget)
        self.items = items
        
    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItems(self.items)
        combo.currentIndexChanged.connect(self.currentIndexChanged)
        return combo
        
    def setEditorData(self, editor, index):
        editor.blockSignals(True)
        editor.setCurrentIndex(int(index.model().data(index)))
        editor.blockSignals(False)
        
    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentIndex())
        
    @pyqtSlot()
    def currentIndexChanged(self):
        self.commitData.emit(self.sender())
        
class App(QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self) 
        self.DepositionButton.clicked.connect(self.deposition)
        self.actionenter.triggered.connect(self.deposition)
        self.update_model_Button.clicked.connect(self.update_model)
        self.save_settings_Button.clicked.connect(self.save_settings)
        self.open_settings_Button.clicked.connect(self.open_settings)
        self.save_path = 'saves/'
        settings = read_excel(self.save_path+'settings.xlsx', index_col=0)
        self.update_settings(settings)
        self.set_delegates(self.table_settings)
        self.R_Slider.valueChanged.connect(self.set_R_slider)
        self.k_Slider.valueChanged.connect(self.set_k_slider)
        self.NR_Slider.valueChanged.connect(self.set_NR_slider)
        self.update_model()
        self.R_disp.editingFinished.connect(self.set_R_line)
        self.k_disp.editingFinished.connect(self.set_k_line)
        self.NR_disp.editingFinished.connect(self.set_NR_line)
        self.set_R(25)
        self.set_k(1.5)
        self.set_NR(1)
        
    def set_delegates(self, table_view):
        l = 0
        while self.model_settings.index(l,self.settings.index_type).isValid():
           l+=1
        for i in range(l):
            items = self.model_settings.itemData(self.model_settings.index(i,self.settings.index_type))[self.settings.index_type-1]
            if 'cases' in items:
                d = {}
                exec(items, d) # cases = [..., ..., ...]
                items = d['cases']
                for j, item in enumerate(items):
                    items[j] = str(item)
                table_view.setItemDelegateForRow(i, DropboxDelegate(table_view, items))
        
    def update_settings(self, settings):
        self.settings = Settings(array(settings, dtype=object))
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
        
    def save_settings(self):
        name, flag = QInputDialog.getText(self, 'Input Dialog',
            'File name:')
        if flag:
            self.update_model()
            self.settings.save(self.save_path+name)
        
    def open_settings(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', os.getcwd()+'/'+self.save_path+'settings.xlsx')[0]
        settings = read_excel(fname, index_col=0)
        self.update_settings(settings)
        self.update_model()
        
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
    
def main():
    print('app = main')
    app = QApplication(sys.argv)  
    window = App() 
    window.show()  # Показываем окно
    sys.exit(app.exec_())
    
if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    import sys
    main()  # то запускаем функцию main()