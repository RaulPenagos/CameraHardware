import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize 
from statsmodels.base.model import GenericLikelihoodModel




class CircleFit(GenericLikelihoodModel):
    """
    Clase Likelihood para crear modelos a partir de los puntos (x, y)
    del borde de un círculo y obtener mediante una minimización la posición
    del centro (x_c, y_c) y el radio r del círculo que mejor se ajusta a los datos.
    """

    def __init__(self, exog, endog, **kwds):
        """
        exog (array):  X, coordenadas x de los puntos del borde del círculo
        endog (array): Y, coordenadas y de los puntos del borde del círculo
        """
        self.n = int(len(exog))
        
        self.exog = np.asarray(exog)
        self.endog = np.asarray(endog)
        print(self.exog)
        print(self.endog)
        # Se dan valores iniciales razonables a los parámetros-----------------
        # self.a = np.mean(exog)  # Posición centro en x
        # self.b = np.mean(endog)  # Posición centro en y
        self.a = 400 # Posición centro en x
        self.b = -200
        self.r = (np.max([(max(self.exog)-min(self.exog)), (max(self.endog)-min(self.endog))]))  # Radio círculo

        super(CircleFit, self).__init__(endog, exog, **kwds)  


    def loglike(self, params):
        #  Se actualizan los parámetros
        self.a = params[0]
        self.b = params[1]
        self.r = params[2]

        chi2 = 0.0

        for i in range(0, self.n):
            chi2 += np.abs(((self.exog[i]-self.a)**2 + (self.endog[i]-self.b)**2 - self.r**2))
        print('CHI2: ', chi2)

        return -chi2

    def fit(self, start_params=None, method='nm', maxiter=10000, **kwargs):

        if start_params is None:
            start_params =  [self.a, self.b, self.r]
        return super(CircleFit, self).fit(start_params=start_params, method=method, maxiter=maxiter, **kwargs)
    




def simular_foto_crop_circulo(nn = 1024):

    n = nn

    l = 10
    R = 8

    X, Y = np.meshgrid(np.linspace(-l,l,n), np.linspace(-l,l,n))
    objeto = 1*(X**2 + Y**2 < R**2)

    # Hago el crop
    # Imagen = objeto[150:400, 0:150]
    Imagen = objeto[0:750, 0:150]
    # Imagen = objeto[0:400, 0:550]

    plt.figure(figsize = [4, 10])
    plt.subplot(1,2,1)
    plt.imshow(objeto, cmap = 'gray')
    plt.subplot(1,2,2)
    plt.imshow(Imagen, cmap = 'gray')

    return objeto, Imagen


def search_border(fig):
    # Defino frontera
    y_max , x_max = np.asarray(fig.shape) - 1

    # Desplazamientos en las 4 direcciones principales
    up    = np.roll(fig, shift=-1, axis=0)
    down  = np.roll(fig, shift=1, axis=0)
    left  = np.roll(fig, shift=1, axis=1)
    right = np.roll(fig, shift=-1, axis=1)

    # Detectar bordes: puntos donde hay un 1 y algún vecino es 0
    border_mask = (fig == 1) & ((up == 0) | (down == 0) | (left == 0) | (right == 0))

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

    return x, y


def main():

    n = 1024
    objeto, fig = simular_foto_crop_circulo(n)

    x, y = search_border(fig)

    cal = CircleFit(x, y)

    results = cal.fit()    

    print(results.summary())

    print(results.params)

    a, b, r = results.params

    plt.scatter(a,b, s = 30, c = 'r')   
    plt.scatter(int(n/2), int(n/2), s = 20, c = 'b')
    plt.imshow(objeto, cmap = 'gray')   
    plt.show()




if __name__ == '__main__':

    main()
