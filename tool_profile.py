from PyQt5.QtCore import (Qt, QSortFilterProxyModel, pyqtSignal, pyqtSlot, 
QThread, QModelIndex, QAbstractTableModel, QVariant, QPoint)
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QInputDialog,
    QFileDialog,
    QAbstractItemView,
    QErrorMessage,
    QMessageBox,
    QShortcut,
    QDialog,
    QLabel
)
import os
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
import design_profile
import numpy as np
from math import ceil, floor

def round_to_1(x):
   return round(x, -int(floor(np.log10(abs(x)))))

class Profile_app(QMainWindow, design_profile.Ui_Profile):

    def __init__(self, parent, x0, y0, h0, x, y, h):
        super().__init__(parent)
        self.setupUi(self)
        self.parent = parent
        self.x0 = x0
        self.h0 = h0
        self.prec_R = 1
        self.prec_h = max(1,-int(floor(np.log10(h0.max()-h0.min()))))
        self.annotation_y = h0.mean()
        R_min, R_max = parent.model.R_bounds
        R = R_min
        self.r = parent.model.substrate_radius
        _R_min = parent.model.holder_inner_radius
        _R_max = parent.model.holder_outer_radius
        self.convert_to_slider = (lambda x: int(round(99*(x-R_min)/(R_max-R_min))))
        self.convert_from_slider = (lambda n: R_min+(R_max-R_min)*n/99)
        self.in_slider_R.valueChanged.connect(self.plot)
        self.actionExport_data.triggered.connect(self.export_data)
        fig = Figure()
        self.canvas = FigureCanvas(fig)
        self.plot_layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self.plot_widget)
        self.plot_layout.addWidget(self.toolbar)
        self.ax1 = self.canvas.figure.add_subplot(121)
        self.ax2 = self.canvas.figure.add_subplot(122)
        ax = self.ax1
        im = ax.contourf(x, y, h)
        self.sub_x = np.array(parent.model.substrate_rect_x)
        self.sub_y = np.array(parent.model.substrate_rect_y)
        self.sub_ref = ax.plot(self.sub_x+R, self.sub_y, color='black')
        ax.set_xlabel('$x, mm$')
        ax.set_ylabel('$y, mm$')
        ax.set_title('Single rotation')
        clb=ax.figure.colorbar(im, ax=ax)
        clb.ax.set_title('$h, \\%$')
        ax = self.ax2
        ax.plot(x0, h0)
        ax.axvline(_R_min, linestyle='--', color='gray')
        ax.axvline(_R_max, linestyle='--', color='gray')
        self.left_ref = ax.axvline(R-self.r, color='black')
        self.right_ref = ax.axvline(R+self.r, color='black')
        self.annotation_ref = ax.annotate(f'$R = {round(R, self.prec_R)}mm$\n$\Delta h = {round(self.variation(R),self.prec_h)}$%', (R, self.annotation_y))
        ax.set_xlabel('$r, mm$')
        ax.set_ylabel('$h, %$')
        ax.set_title('Cross section')
        self.canvas.figure.tight_layout()
        self.canvas.draw()
        
        
    @pyqtSlot(int)  
    def plot(self, n):
        R = self.convert_from_slider(n)
        self.left_ref.remove() 
        self.right_ref.remove() 
        ax = self.ax2
        self.left_ref = ax.axvline(R-self.r, color='black')
        self.right_ref = ax.axvline(R+self.r, color='black')
        self.sub_ref[0].set_xdata(self.sub_x+R)
        self.annotation_ref.remove()
        self.annotation_ref = ax.annotate(f'$R = {round(R, self.prec_R)}mm$\n$\Delta h = {round(self.variation(R),self.prec_h)}$%', (R, self.annotation_y))
        self.canvas.draw()
        
    def variation(self, R):
        R1 = R-self.r
        R2 = R+self.r
        i1 = (self.x0 < R1).sum()
        i2 = (self.x0 < R2).sum()
        h_min = self.h0[i1:i2].min()
        h_max = self.h0[i1:i2].max()
        return h_max-h_min
        
    def export_data(self):
        fname, flag = QInputDialog.getText(self, 'Input Dialog',
            'File name:')
        if flag:
            data = np.transpose(np.array([self.x0,self.h0]))
            np.savetxt(fname+'.txt', data, header='#Profile cross-section\nr[mm] h[nm]')
        
        