# import sys
# #
# # # Setting the Qt bindings for QtPy
# # import os
# # os.environ["QT_API"] = "pyqt5"
# #
# # from qtpy import QtWidgets
# #
# # import numpy as np
# #
# # import pyvista as pv
# # from pyvistaqt import QtInteractor, MainWindow
# #
# # class MyMainWindow(MainWindow):
# #
# #     def __init__(self, parent=None, show=True):
# #         QtWidgets.QMainWindow.__init__(self, parent)
# #
# #         # create the frame
# #         self.frame = QtWidgets.QFrame()
# #         vlayout = QtWidgets.QVBoxLayout()
# #
# #         # add the pyvista interactor object
# #         self.plotter = QtInteractor(self.frame)
# #         vlayout.addWidget(self.plotter.interactor)
# #         self.signal_close.connect(self.plotter.close)
# #
# #         self.frame.setLayout(vlayout)
# #         self.setCentralWidget(self.frame)
# #
# #         # simple menu to demo functions
# #         mainMenu = self.menuBar()
# #         fileMenu = mainMenu.addMenu('File')
# #         exitButton = QtWidgets.QAction('Exit', self)
# #         exitButton.setShortcut('Ctrl+Q')
# #         exitButton.triggered.connect(self.close)
# #         fileMenu.addAction(exitButton)
# #
# #         # allow adding a sphere
# #         meshMenu = mainMenu.addMenu('Mesh')
# #         self.add_sphere_action = QtWidgets.QAction('Add Sphere', self)
# #         self.add_sphere_action.triggered.connect(self.add_sphere)
# #         meshMenu.addAction(self.add_sphere_action)
# #
# #         if show:
# #             self.show()
# #
# #     def add_sphere(self):
# #         """ add a sphere to the pyqt frame """
# #         sphere = pv.Sphere()
# #         self.plotter.add_mesh(sphere, show_edges=True)
# #         self.plotter.reset_camera()
# #
# #
# # if __name__ == '__main__':
# #     app = QtWidgets.QApplication(sys.argv)
# #     window = MyMainWindow()
# #     sys.exit(app.exec_())









# # -*- coding: utf-8 -*-
# """
# Various methods of drawing scrolling plots.
# """
# #import initExample  ## Add path to library (just for examples; you do not need this)
#
# import pyqtgraph as pg
# from pyqtgraph.Qt import QtCore, QtGui
# import numpy as np
#
# win = pg.GraphicsLayoutWidget(show=True)
# win.setWindowTitle('pyqtgraph example: Scrolling Plots')
#
# # 1) Simplest approach -- update data in the array such that plot appears to scroll
# #    In these examples, the array size is fixed.
# p1 = win.addPlot()
# p2 = win.addPlot()
# data1 = np.random.normal(size=300)
# curve1 = p1.plot(data1)
# curve2 = p2.plot(data1)
# ptr1 = 0
#
#
# def update1():
#     global data1, ptr1
#     data1[:-1] = data1[1:]  # shift data in the array one sample left
#     # (see also: np.roll)
#     data1[-1] = np.random.normal()
#     curve1.setData(data1)
#
#     ptr1 += 1
#     curve2.setData(data1)
#     curve2.setPos(ptr1, 0)
#
#
# # 2) Allow data to accumulate. In these examples, the array doubles in length
# #    whenever it is full.
# win.nextRow()
# p3 = win.addPlot()
# p4 = win.addPlot()
# # Use automatic downsampling and clipping to reduce the drawing load
# p3.setDownsampling(mode='peak')
# p4.setDownsampling(mode='peak')
# p3.setClipToView(True)
# p4.setClipToView(True)
# p3.setRange(xRange=[-100, 0])
# p3.setLimits(xMax=0)
# curve3 = p3.plot()
# curve4 = p4.plot()
#
# data3 = np.empty(100)
# ptr3 = 0
#
#
# def update2():
#     global data3, ptr3
#     data3[ptr3] = np.random.normal()
#     ptr3 += 1
#     if ptr3 >= data3.shape[0]:
#         tmp = data3
#         data3 = np.empty(data3.shape[0] * 2)
#         data3[:tmp.shape[0]] = tmp
#     curve3.setData(data3[:ptr3])
#     curve3.setPos(-ptr3, 0)
#     curve4.setData(data3[:ptr3])
#
#
# # 3) Plot in chunks, adding one new plot curve for every 100 samples
# chunkSize = 100
# # Remove chunks after we have 10
# maxChunks = 10
# startTime = pg.ptime.time()
# win.nextRow()
# p5 = win.addPlot(colspan=2)
# # print(type(p5))  # <class 'pyqtgraph.graphicsItems.PlotItem.PlotItem.PlotItem'>
# print("type: ", type(p5.plot()))
# p5.setLabel('bottom', 'Time', 's')
# p5.setXRange(-10, 0)
# curves = []
# data5 = np.empty((chunkSize + 1, 2))
# ptr5 = 0
#
#
#
# def update3():
#     global p5, data5, ptr5, curves
#     now = pg.ptime.time()
#     for c in curves:
#         print(-(now - startTime))
#         c.setPos(-(now - startTime), 0)
#         #c.setPos(-5, 0)
#
#     i = ptr5 % chunkSize
#     if i == 0:
#         curve = p5.plot()
#         curves.append(curve)
#         last = data5[-1]
#         data5 = np.empty((chunkSize + 1, 2))
#         data5[0] = last
#         while len(curves) > maxChunks:
#             c = curves.pop(0)
#             p5.removeItem(c)
#     else:
#         curve = curves[-1]
#     data5[i + 1, 0] = now - startTime
#     data5[i + 1, 1] = np.random.normal()
#     curve.setData(x=data5[:i + 2, 0], y=data5[:i + 2, 1])
#     ptr5 += 1
#
#
# # update all plots
# def update():
#     update1()
#     update2()
#     update3()
#
#
# timer = pg.QtCore.QTimer()
# timer.timeout.connect(update)
# timer.start(50)
#
# ## Start Qt event loop unless running in interactive mode or using pyside.
# if __name__ == '__main__':
#     import sys
#
#     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#         QtGui.QApplication.instance().exec_()










# import numpy as np
#
# class Obj:
#     def __init__(self):
#         self.data_array = np.array([[[1,1,1],[2,2,2],[3,3,3],[4,4,4]], [[5,5,5],[6,6,6],[7,7,7],[8,8,8]], [[9,9,9],[10,10,10],[11,11,11],[12,12,12]]])
#         self.count_array = np.array([[[1,1,1],[2,2,2],[3,3,3],[4,4,4]], [[5,5,5],[6,6,6],[7,7,7],[8,8,8]], [[9,9,9],[10,10,10],[11,11,11],[12,12,12]]])
#         self.lat = 1.1
#         self.lon = 2.2
#         self.timestamp = 12.00
#
# obj1 = Obj()
# obj2 = Obj()
# obj3 = Obj()
#
# print(obj1.lat)
#
# obj_list = [obj1, obj2, obj3]
# obj_array = np.array(obj_list)
#
# print(obj_array[0])
# print(obj_array[0].lon)
#
# print(type(obj_array[0]))
# print(type(obj_array[:]))



# print(obj_array[:][:].data_array)
# print(obj_array[:].data_count)
# print(obj_array[:].data_array + obj_array[:].data_count)

# import pyqtgraph.examples
# pyqtgraph.examples.run()

if __name__ == "__main__":
    x = 0
    while True:
        for i in range(10):
            x += 1
        if x > 5:
            break
    print("after true")