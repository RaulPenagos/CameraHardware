"""
Script defining Camera and Image classes. Enables the use of IDS cameras, and processing of 
taken images. Uses CircleCompleter in order to calculate the center of circles given an image
of just a portion.

Author: Raul Penagos
Date: Feb 13th, 2025
"""

import ids_peak.ids_peak as ids_peak
import ids_peak_ipl.ids_peak_ipl as ids_ipl
import ids_peak.ids_peak_ipl_extension as ids_ipl_extension
import numpy as np
from matplotlib import pyplot as plt
import datetime as dt


# sys.path.append('/CameraControl/test')
# from test.makeMeasurements import *
from CircleCompleter import CircleFit
from matplotlib.image import imread
import cv2

import os # path to the project, used in Image.save()



class Image:
    """
    Class Image, helps in foramting and processing pictures taken with an IDS camera.
    Methods of the class:
    save()    display()   binarize()  soften()    find_cm()   search_border()  circle_treatment()

    Para facilitar la minimaización definir unos valores iniciales del centro acorde con lo esperado
    """
    def __init__(self, matrix):
        self.image = np.copy(matrix[:,:,0])
        self.image_original = np.copy(matrix)
        self.timestamp = dt.datetime.now()
        self.file_name = f'img/tests/img_{self.timestamp.strftime("%Y%m%d_%H%M%S")}.png'

        dirname = os.path.dirname(__file__)  # Edit where to save the img
        self.abs_filename = os.path.join(dirname, self.file_name)

        self.cm = (None, None)


    def save(self):
        """
        Save the image to the file name given by the Image's time stamp.
        Make sure the folder exists.
        """
        plt.figure(figsize = (5,5))
        plt.imshow(self.image)
        
        plt.savefig(self.abs_filename)

    def display(self, save = False):
        """
        Show and optionally save the image on screen.
        """
        plt.close("all")
        plt.figure(figsize = (5,5))
        plt.imshow(self.image, cmap = 'gray')

        if save: 
            plt.savefig(self.abs_filename)

        if self.cm != (None, None):
            plt.plot(self.cm[0], self.cm[1], 'or')

        plt.show()

    def binarize(self):
        """
        Binarize the image to (0, 255) grey scale (black and white).
        """
        self.image = np.where(self.image > self.image.max()/2, 255, 0)
        return self
        
    def soften(self, m = 8 ):
        """
        Soften filter smoothens the image by taking and average of a m shape
        square kernel.
        """
        # Average filter
        for i in range(0+m, self.image.shape[0] - m):
            for j in range(0+m, self.image.shape[1] - m):
                self.image[i,j] = np.mean(self.image[i-m:i+m, j-m:j+m])
        return self
    
    def find_cm(self):
        """
        Given a binarized picture, it subtracts the center of fiducials (white) 
        on a black Background.
        """
        # Average of white Pixels, for binarized pictures 
        yy, xx = np.where(self.image > 0)
        # self.cm = np.array([xx, yy])
        self.cm = (xx.mean(), yy.mean()) if xx.size > 0 else (None, None)
             
    def search_border(self):
        """
        Busca los pixels de frontera en una imagen binarizada, aquellos que estén rodeados 
        por pixels de distinto color. (up, down, right, left)
        Si hago una cruz con 'brazos mas largos' y exijo que dos sean negros y dos blancos
        podría mejorar y quitarme tantos puntos de ruido
        """
        # Defino frontera
        y_max , x_max = np.asarray(self.image.shape) - 1

        # Desplazamientos en las 4 direcciones principales
        up    = np.roll(self.image, shift=-1, axis=0)
        down  = np.roll(self.image, shift=1, axis=0)
        left  = np.roll(self.image, shift=1, axis=1)
        right = np.roll(self.image, shift=-1, axis=1)

        print(self.image.max())
        print(self.image.min()) 

        # Detectar bordes: puntos donde hay un 1 y algún vecino es 0
        # border_mask = (self.image == 1) & ((up == 0) | (down == 0) | (left == 0) | (right == 0))
        border_mask = (self.image == 0) & ((up > 0) | (down > 0) | (left > 0) | (right > 0))

        # Obtener coordenadas de los bordes
        y, x = np.where(border_mask)

        # Quito elementos de los bordes de imagen
        index_x = np.copy([i for i, xx in enumerate(x) if (xx <= 1 or xx >= x_max)])
        if len(index_x) != 0:
            y = np.delete(y, index_x)
            x = np.delete(x, index_x)

        index_y = np.copy([i for i, xx in enumerate(y) if (xx <= 1 or xx >= y_max)])
        if len(index_y) != 0:
            y = np.delete(y, index_y)
            x = np.delete(x, index_y)

        plt.plot(x, y, 'ro') 
        plt.imshow(self.image, cmap = 'gray')
        plt.show()

        return x, y
    

    def canny(self):
        """
        No funciona bien con imagenes binarizadas, el algoritmo no lo permite
        """

        # fig = cv2.imread('./biblia.jpg')[: , :, 0]
        # fig = imread('./biblia.jpg')[: , :, 0]
        fig = self.image

        bordeCanny = cv2.Canny(fig, 100, 200)

        cv2.imshow('Original', fig)
        cv2.imshow('Canny', bordeCanny)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        x, y = np.array([]), np.array([])
        for i,f in enumerate(bordeCanny):
            for j,px in enumerate(f):
                if px != 0:
                    x = np.append(x, j)
                    y = np.append(y, np.abs(i-800))

        plt.plot(x,y , 'or')

        # plt.gca().invert_yaxis()
        plt.show()

        return x,y

    
    def circle_treatment(self, show = True):
        """
        Given the coordinates x,y of a circles border, or segment of its border
        It will minimize through a Likelihood function the center (a,b) of the 
        circle and its radius (r).
        Returns: 
            a: posición centro en x
            b: posición centro en y
        """
        x, y = self.search_border()

        cal = CircleFit(x, y)

        results = cal.fit()    

        print(results.summary())

        print(results.params)

        a, b, r = results.params

        if show:
            plt.imshow(self.image, cmap = 'gray')   
            plt.scatter(a,b, s = 30, c = 'r')   
            plt.show()


        return a,b 



class Camera:
    """
    Class that enables the conection with an IDS industrial camera by creating an instance of it.
    Enables changing exposure_time, take images process and save them as Image instances.
    """
    def __init__(self):

        self.device_descriptors = None
        self.device = None
        self.remote_device_nodemap = None
        self.datastream = None
        self.exposure_time_seg = 1/250
        
        self.search_device()
        self.name = self.device_descriptor.DisplayName()
        self.open_device()

        self.image = None
        

    def search_device(self):
        """
        Searches for devices compatible with IDS industrial cameras
        """
        try:
            ids_peak.Library.Close()
            ids_peak.Library.Initialize()
            device_manager = ids_peak.DeviceManager.Instance()
            device_manager.Update()
            self.device_descriptors = device_manager.Devices()

            print("Found Devices: " + str(len(self.device_descriptors)))

            for self.device_descriptor in self.device_descriptors:
                print(self.device_descriptor.DisplayName())

            return self
        except Exception as e:
            print('ERR:' + str(e))
        finally:
            ids_peak.Library.Close()


    def open_device(self):
        """
        Opens available devices.
        Will give an error if the devices are already in use
        """
        try:
            self.device = self.device_descriptors[0].OpenDevice(ids_peak.DeviceAccessType_Control)
            print("Opened Device: " + self.device.DisplayName())
            self.remote_device_nodemap = self.device.RemoteDevice().NodeMaps()[0]

            # Set Software trigger: Single frame acquisition
            self.remote_device_nodemap.FindNode("TriggerSelector").SetCurrentEntry("ExposureStart")
            self.remote_device_nodemap.FindNode("TriggerSource").SetCurrentEntry("Software")
            self.remote_device_nodemap.FindNode("TriggerMode").SetCurrentEntry("On")

            return self
        except Exception as e:
            print('No device is free and available. ERR:' + str(e))
        finally:
            ids_peak.Library.Close()

    def start_acquisition(self):
        """
        Starts acquisition time, during this time Images can be taken
        """
        try:
            self.datastream = self.device.DataStreams()[0].OpenDataStream()
            payload_size = self.remote_device_nodemap.FindNode("PayloadSize").Value()
            for i in range(self.datastream.NumBuffersAnnouncedMinRequired()):
                buffer = self.datastream.AllocAndAnnounceBuffer(payload_size)
                self.datastream.QueueBuffer(buffer)

            self.datastream.StartAcquisition()
            self.remote_device_nodemap.FindNode("AcquisitionStart").Execute()
            self.remote_device_nodemap.FindNode("AcquisitionStart").WaitUntilDone()

            return self
        except Exception as e:
            print('No device is free and available. ERR:' + str(e))
        finally:
            ids_peak.Library.Close()

    def set_exposure(self, exposure_time_seg = 1/250):
        """
        Sets exposure time for the capture
        """
        try: 
            self.exposure_time_seg = exposure_time_seg
            exposure_time_microseg = exposure_time_seg * 1e6
            self.remote_device_nodemap.FindNode("ExposureTime").SetValue(exposure_time_microseg) # in microseconds  # in microseconds

            return self
        except Exception as e:
            print('No device is free and available. ERR:' + str(e))
        finally:
            ids_peak.Library.Close()

    def get_image(self):
        """
        Triggers the camera and gets a picture of type Image
        """
        try:
            # trigger image
            self.remote_device_nodemap.FindNode("TriggerSoftware").Execute()
            buffer = self.datastream.WaitForFinishedBuffer(1000)

            # convert to RGB
            raw_image = ids_ipl_extension.BufferToImage(buffer)
            # for Peak version 2.0.1 and lower, use this function instead of the previous line:
            #raw_image = ids_ipl.Image_CreateFromSizeAndBuffer(buffer.PixelFormat(), buffer.BasePtr(), buffer.Size(), buffer.Width(), buffer.Height())
            color_image = raw_image.ConvertTo(ids_ipl.PixelFormatName_RGB8)
            self.datastream.QueueBuffer(buffer)

            self.image = Image(color_image.get_numpy_3D())
        except Exception as e:
            print('No device is free and available. ERR:' + str(e))
        finally:
            ids_peak.Library.Close()

    def close_device(self):
        """
        Closes the libraries, seting free the device in use. 
        """
        ids_peak.Library.Close()

    def auto_exposure_get_image(self, gray_pallete = 50):
        """
        Sets automatically exposure, no matters the extern ilumination conditions, 
        given by the light source.
        Computes the average luminance of the frame and compares to a gray_pallete value.
        Args:
            gray_pallete: Value to compare with the average luminance. 
            --Recommended values:--
            > Fiducials = 50
            > Calibration Dots = ... 
        https://stackoverflow.com/questions/73611185/automatic-shutter-speed-adjustment-feedback-algorithm-based-on-images

        """
        try:
            self.set_exposure(self.exposure_time_seg)
            self.get_image()

            L1 = np.mean(self.image.image) # Compute the average luminance of the current frame 
            print(L1)

            L2 = gray_pallete # Gray Card reference

            a = 0.5  # a = 0.5 parameter is tuneable

            #  Compute exposure Value
            # EV = np.log2(L1)/np.log2(L2)
            self.set_exposure(self.exposure_time_seg*(120 / L1) ** a)  

            while np.abs(L1-L2) > 5:
                self.get_image() 
                L1 = np.mean(self.image.image)
                self.set_exposure(self.exposure_time_seg*(L2 / L1) ** a) 
            self.get_image() 
            self.image.display()
        except Exception as e:
            print('No device is free and available. ERR:' + str(e))
        finally:
            ids_peak.Library.Close()


    def fiducial_protocole(self):
        print('ToDo')
        # Como utilizar la clase Image y sus atributos desde aquí


def test1():
     # ----DEBUGGING---- Funciones clase Camera ------------

    camera = Camera()

    camera.start_acquisition().set_exposure()

    camera.get_image()

    picture = camera.image

    # picture.save()

    picture.display()

    camera.close_device()



def test2():
    # ----DEBUGGING---- Funciones clase Image ------------

    importar = imread('./img/fiducial_test.png')

    my_example = Image(importar[:,:,0])

    # my_example.display()
    # my_example.soften().display()
    my_example.binarize()

    my_example.find_cm()

    # my_example.display()
    my_example.soften(2).binarize().display()


def test3():
    # ----DEBUGGING---- Funciones Circulos ------------

    importar = imread('./img/fiducial_test5.png')

    my_example = Image(importar[:,:,0])

    # my_example.display()

    my_example.binarize()

    my_example.search_border()

    my_example.circle_treatment()
    


def test4():
    # ----DEBUGGING---- Auto Exposure  ------------

    camera = Camera()

    camera.start_acquisition().set_exposure()

    camera.set_exposure(1/2000)

    camera.get_image()

    camera.image.display()

    camera.auto_exposure_get_image()

    camera.close_device()

def test5():
    # ----DEBUGGING---- Funciones Circulos con tornillo REAL ------------

    camera = Camera()

    camera.start_acquisition().set_exposure(1/250)

    camera.get_image()

    picture = camera.image

    picture.display()

    picture.soften().display()

    picture.binarize().display()

    picture.search_border()

    picture.circle_treatment()

    camera.close_device()

def test_6():
    # ----DEBUGGING---- Funciones Canny con foto de Tornillo REAL ------------
    #  TO DO, test y probar a usar filtro soften si salen demasiados bordes
    fig = Image(cv2.imread('./hole1.png'))
    # fig = Image(cv2.imread('./fiducial_test.png'))

    fig.soften(2)

    bordeCanny = cv2.Canny(fig.image, 100, 200)

    cv2.imshow('Original', fig.image)
    cv2.imshow('Canny', bordeCanny)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



def main():

    test_6()
      
   



if __name__ == "__main__":

    main()

    

    



    

