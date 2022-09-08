### para mi librería 
import math
import numpy as np
import sys
import pickle

def aproxex(x,n):
    """
    Está función aproxima el valor de e^x utilizando una serie de taylor.
    está función recibe como parámetros obligarios de entrada el valor de x para aproximar e^x, 
    así como el orden n al que se aproximará. 
    Función realizada por Javier Francisco Castilla Jiménez.
    
    """
    try:
        aprox=0
        for k in range(n+1):
            ex=(np.power(x,k))/(math.factorial(k))
            aprox=np.add(aprox,ex)
        return aprox
    except TypeError:
        if (type(n)==str) or (type(x)==str):
            print('El error fue debido a que introdujo un valor no númerico para alguno de los parámetros')
            print('el error fue:',sys.exc_info()[0])
        elif (type(n)==float):
            print('el parametro n debe ser un entero no un float')
    else:
        print('el error fue:',sys.exc_info()[0])



def randompoints(arreglo,xpuntos):
    """
    Esta función sirve para generar vectores/puntos aleatorios entre los valores extremos de una matriz de referencia cuyos 
    renglones son tomados como puntos aleatorios. Dicho de otra forma cada elemento de los nuevos vectores aleatorios estará 
    entre los valores extremos de cada columna de la matriz original. Cada nuevo vector aleatorio tiene la misma cantidad de 
    columnas que tiene la matriz de referencia.Los nuevos vectores aleatorios se almacenan como los renglones de una nueva matriz.
    Tiene dos parametros ambos obligatorios, la matriz de referencia (arreglo) y el número de nuevos puntos/vectores(xpuntos) 
    que se quiere generar.
    
    """
    minimos=np.max(arreglo,axis=0) #usamos np.max y especificamos axis=0 para indicar que queremos los maximos de las columnas. observemos que evitamos el uso de ciclo for para optimizar el código.
    maximos=np.min(arreglo,axis=0) #usamos np.min y especificamos axis=0 para indicar que queremos los minimos de las columnas.
    ptrd=[]
    for i in range (xpuntos):#creamos x puntos aleatorios entre nuestro valores extremos
        ptrd.append(np.random.uniform(minimos,maximos))
    ptrdm=np.array(ptrd)    
    return ptrdm
    
###  para funciones
def norma(x,y):
    """
    Esta función sirve para calcular la norma o distancia euclidiana entre dos puntos (vectores) de un espacio euclidio
    los parametros de entrada son los dos vectores entre los que se desea calcular la norma.
    """
    norma=np.sqrt((np.sum((x-y)**2,axis=1)))
    return norma

def guardar(arreglo,npy=False,pic=False):
    """
    Esta función recibe un arreglo numpy y lo guarda como csv, una ves corriendo le pregunta al usuario el nombre que con
    el que se guardara el archivo(no es necesario escribir la extensión esta se agrega automáticamente).Adicionalmente la función  puede también  guardar el arreglo como npy y como pickle codificado
    en binario, sin embargo por defecto no lo guardará en pickle ni en npy a menos que lo específique el usuario.Si el usuario 
    decide guardar el archivo en un formato adicional a csv el archivo se llamara igual en todos los formatos, solo cambiará 
    la extensión.El único parámetro obligatorio es el arreglo que se guardará. Tiene dos parámetro opcionales npy y pic pero 
    por defecto están en false. npy=True támbien guardará el archivo con extensión npy. pic=True También guardará el archivo 
    como pickle codificado en binario. La función no devuelve nada.Solamente realiza el proceso de guardado.
    
    """
    x=input('ingrese el nombre del archivo')
    np.savetxt(x+'.csv',arreglo,delimiter=',') #guardamos el archivo como csv
    if npy==True: # si el usuario indica npy=True
        np.save(x+'.npy',arreglo)#guardamos el archivo como npy
    if pic==True: # si el usuario indica pic=True
        pickle_out= open(x+'.pickle','wb')#guaradamos el archivo como pickle
        pickle.dump(arreglo, pickle_out)
        pickle_out.close()
        
def clasificacion(distancias):
    """
    Esta función resuelve una versión generalizada del tercer inciso del ejercicio 2 de la Tarea 3 de programación.Está función
    genera un vector de n elementos. Cada elemento indica a cuál de los puntos está más cerca cada vector/renglón de la matriz 
    original. Si hay un empate (es decir un vector está a la misma distancia de dos puntos) resuelvé con una moneda al aire (en
    este caso radom choice con una distribución uniforme de probabilidad para cada punto empatado)

    """
    indicesmin=np.argmin(distancias,axis=1) #indices de los minimos fila a fila, es decir el indice de la columna en la que esta
    # el minimo en cada fila  # en ese indice de cada columna esta el valor de distancia minima (el punto más cercano)
    vectormin=np.min(distancias, axis=1,keepdims=True) #este vector contiene los valores los 
    #valores minimos de cada renglon
    resta=distancias-vectormin #en los lugares donde estaba el minimo quedara un cero pero tambien en los lugares donde
    #haya un empate quedará un cero  es entonces si en alguna fila hay más de un cero hubo al menos un empate y generamos un 
    #numero aleatorio para desempatar y sino hubo empates la función regresa el arreglo con los indices de los minimos 
    empates=np.apply_along_axis(sum,1,resta==0)
    cerca=[] #lista vacía para guardar el juicio de cercania 
    for i in range(indicesmin.shape[0]):# en el rango del número de puntos que tiene la matriz de minimos
        if empates[i]==1: # si la suma por renglón es igual a 1 es porque no hay empates
            cerca.append([indicesmin[i]])# agregamos el indice del mínimo correspondiente 
        else:# si la suma por renglón no es igual a 1 es porque hay un empate
            result=np.where(resta == 0) # localizamos las coordenadas empatadas
            choices=result[0] #p es un array con la coordenadas de los empates
            eleccion=np.random.choice(choices, 1) # tiramos una moneda al aire
            cerca.append([eleccion]) # agregamos el indice de la elección al azar correspondiente 
    clasificacion=np.array(cerca)# hacemos un np array con la clasificación
    return  clasificacion #devuelve la clasificación

    
def ejercicio2(arreglo,xpuntos,save=False):
    """
    Esta función realiza el procedimiento descrito en el ejercicio 2 de la Tarea 3 de programación.La función 
    debe recibir como parámetros obligatorios: el arreglo de numpy,el número de puntos a generar.Como parámetros 
    opcionales la función guarda o no el archivo como csv, pickle y npy. Por defecto no lo guardará a menos que lo 
    específique el usuario. En caso de que se acepte guardar, la función solicita al usuario el nombre del 
    archivo que se generará. La función devuelve la clasificación realizada.
    """
    puntos=randompoints(arreglo,xpuntos)
    listad=[]# declaramos una lista que guardará los arreglos con las distancias
    for i in range(puntos.shape[0]):#iteramos en el rango de los renglones de nuestro arreglo de numeros aleatorios
        distancia=norma(arreglo,puntos[i]) #distancia es un array con las distancias de cada punto del arreglo de referencia a uno de mis puntos aleatorios
        listad.append(distancia) # agregamos a una lista nuestros arreglos de distancias # usamos una lista por la facilidad de agragrle elmentos durante un ciclo
    
    distancias=np.array(listad) #convertimos la lista con los arreglos de distancias a un arreglo pero en este arreglo los renglones son lo queremos que sean columnas y al revés
    new=(distancias.shape[1],distancias.shape[0]) # creamos una tupla con las coordenadas invertidas  
    distancias=np.reshape(distancias,new) #cambiamos las orientación del array distancias de forma que ahora es una matriz de dimension NxK, donde N corresponde al numero de puntos a los que medí la distancia euclidiana  y K es el numero de puntos aleatorios que cree. y cada elemento es la distancia euclidiana calculada
    clasif=clasificacion(distancias)#clasificamos las distancias
    if save==True: # si  el usuario indica que si quiere guardar la clasificación
        guardar(clasif,npy=True,pic=True)# se guarda la clasificación en csv,npy y pickle #se le solicita el nombre al usuario
        
    return clasif # return devuelve la clasificación de las distancias 

def newcentroid(arreglo,clasificacion):
    """
    Esta función crea un array de centroides tal cómo los solicita el ejercicio 4 de la tarea 3, donde cada centroide esta formado
    con los promedios por columna de las columnas de la matriz de referencia (arreglo original) que se clasificaron en cierta clase.
    Cada nuevo centroide tiene la misma cantidad de columnas que tiene la matriz de referencia(arreglo).Los nuevos centroides se
    almacenan como los renglones de una nueva matriz. Esta función tiene dos parametros ambos obligatorios, la matriz de referencia 
    (arreglo) y la matriz clasificación (clasificacion).
    """
    tiposclas=np.unique(clasificacion) #los valores unicos en los que clasificamos
    centroid=[] # declaro lista vacía par guardar
    for clase in tiposclas: #iteramos cada uno de estas clasificaciones únicas
        i_subclas=np.where(clasificacion==clase)[0] #los indices de clasificaciones donde hay ese tipo de clasificación particular que corresponde a la varible clase
        nuevos=arreglo[i_subclas] #aquí hago un slice de los renglones clasificados en clase 
        c_mean=np.mean(nuevos,axis=0) # saco el promedio por renglones del slice llamado nuevos
        centroid.append(c_mean)#aquí agrego cada vector centroide formado con los promedios por columna a una lista de centroides llamada centroid
    centroides=np.array(centroid) #convierto mi lista de centroides en un array donde cada elemento es un centoide
    return centroides # regresa un array donde cada elemento es un centroide 

def clasificacion_vertar3(arreglo,centroides):
    """
    Esta función clasifica cómo se especifico en el ejercicio 2 de la tarea 3. Pero lo hace con la variante del ejercicio 1 
    de la tarea 4, de forma que esta vez recibe como parametros de entrada la matriz de referencia y el array de centroides
    Está función genera un vector de n elementos. Cada elemento indica a cuál de los puntos/centroides está más cerca cada
    vector/renglón de la matriz original. Si hay un empate (es decir un vector está a la misma distancia de dos puntos) 
    resuelvé con una moneda al aire (en este caso radom choice con una distribución uniforme de probabilidad para cada 
    punto empatado).
    """
    puntos=centroides
    listad=[]# declaramos una lista que guardará los arreglos con las distancias
    for i in range(puntos.shape[0]):#iteramos en el rango de los renglones de nuestro arreglo de numeros aleatorios
        distancia=norma(arreglo,puntos[i]) #distancia es un array con las distancias de cada punto del arreglo de referencia a uno de mis puntos aleatorios
        listad.append(distancia) # agregamos a una lista nuestros arreglos de distancias # usamos una lista por la facilidad de agragrle elmentos durante un ciclo
    
    distancias=np.array(listad) #convertimos la lista con los arreglos de distancias a un arreglo pero en este arreglo los renglones son lo queremos que sean columnas y al revés
    new=(distancias.shape[1],distancias.shape[0]) # creamos una tupla con las coordenadas invertidas  
    distancias=np.reshape(distancias,new) #cambiamos las orientación del array distancias de forma que ahora es una matriz de dimension NxK, donde N corresponde al numero de puntos a los que medí la distancia euclidiana  y K es el numero de puntos aleatorios que cree. y cada elemento es la distancia euclidiana calculada
    clasif=clasificacion(distancias)#clasificamos las distancias
    return clasif # return devuelve la clasificación de las distancias 

def ejercicio2mod1(arreglo,xpuntos,save=False,return_semillas=False):
    """
    Esta función realiza el procedimiento descrito en el ejercicio 2 de la Tarea 3 de programación.La función 
    debe recibir como parámetros obligatorios: el arreglo de numpy,el número de puntos a generar.Como parámetros 
    opcionales la función guarda o no el archivo como csv, pickle y npy. Por defecto no lo guardará a menos que lo 
    específique el usuario. En caso de que se acepte guardar, la función solicita al usuario el nombre del 
    archivo que se generará. La función devuelve la clasificación realizada.
    esta función además de hacer lo anterior devuelve las semillas (los vectores aleatorios generados) si 
    el parametro opcional return_semillas=True
    """
    puntos=randompoints(arreglo,xpuntos)
    listad=[]# declaramos una lista que guardará los arreglos con las distancias
    for i in range(puntos.shape[0]):#iteramos en el rango de los renglones de nuestro arreglo de numeros aleatorios
        distancia=norma(arreglo,puntos[i]) #distancia es un array con las distancias de cada punto del arreglo de referencia a uno de mis puntos aleatorios
        listad.append(distancia) # agregamos a una lista nuestros arreglos de distancias # usamos una lista por la facilidad de agragrle elmentos durante un ciclo
    
    distancias=np.array(listad) #convertimos la lista con los arreglos de distancias a un arreglo pero en este arreglo los renglones son lo queremos que sean columnas y al revés
    new=(distancias.shape[1],distancias.shape[0]) # creamos una tupla con las coordenadas invertidas  
    distancias=np.reshape(distancias,new) #cambiamos las orientación del array distancias de forma que ahora es una matriz de dimension NxK, donde N corresponde al numero de puntos a los que medí la distancia euclidiana  y K es el numero de puntos aleatorios que cree. y cada elemento es la distancia euclidiana calculada
    clasif=clasificacion(distancias)#clasificamos las distancias
    if save==True: # si  el usuario indica que si quiere guardar la clasificación
        guardar(clasif,npy=True,pic=True)# se guarda la clasificación en csv,npy y pickle #se le solicita el nombre al usuario
    if return_semillas==True:
        return clasif,puntos
    else:
        return clasif # return devuelve la clasificación de las distancia

def kmeans(arreglo,nseeds=2,random=True,semillas=None,n_iteraciones=100,tolerancia=1*(10**(-3))):
    """
    Esta función cálcula el realiza el algoritmo k-means de acuerdo a como lo indica el ejercicio 1 de la tarea 4 de programación.
    Devuelve la clasificación y los centroides.Su único parametro obligarorio es el arreglo de referencia. como argumentos de entrada 
    opcionales son el número de iteraciones (default=100) y la tolerancia (default 1e−3) además del tamaño del seed random inicial 
    (que por default es 2). Usa además el argumento opcional random para seleccionar si los centroides iniciales aleatorios o introducidos por el usuario.
    random por default=True es decir por default las semillas son aleatorias a menos que el usuario indique lo contrario. 
    """
    if random==True:
        clasificacion,seeds=ejercicio2mod1(arreglo,nseeds,return_semillas=True)# ejecutamos el ejercicio uno y generamos la primera clasificacion y un arreglo con las semillas
        previous=seeds# centroide previo para la primera iteración el centroide previo son la semillas
    else:
        clasificacion=clasificacion_vertar3(arreglo,semillas)# genero clasificacion en base a mis centroides
        previous=semillas
    for iteracion in range (1,n_iteraciones+1):# iteraciones por default 100
        centroides=newcentroid(arreglo,clasificacion) # genero los centroides pormediando columnas
        clasificacion=clasificacion_vertar3(arreglo,centroides)# genero clasificacion en base a mis centroides
        distanciasC=norma(centroides,previous) # calculo la distancia entre un centroide y el centroide de la it anterior
        previous=centroides # actualizo el valor del centroide previo
        suma=np.sum(distanciasC)
        if suma<=tolerancia:
            break
    return clasificacion,centroides

def kmeans2(arreglo,nseeds=2,random=True,semillas=None,n_iteraciones=100,tolerancia=1*(10**(-3))):
    """
    Esta función cálcula el realiza el algoritmo k-means de acuerdo a como lo indica el ejercicio 1 de la tarea 4 de programación.
    Devuelve la clasificación y los centroides y la iteración en que se quedó.Su único parametro obligarorio es el arreglo de referencia. como argumentos de entrada 
    opcionales son el número de iteraciones (default=100) y la tolerancia (default 1e−3) además del tamaño del seed random inicial 
    (que por default es 2). Usa además el argumento opcional random para seleccionar si los centroides iniciales aleatorios o introducidos por el usuario.
    random por default=True es decir por default las semillas son aleatorias a menos que el usuario indique lo contrario. 
    Esta funcion a diferencia de kmeans también devuelve la iteración en que se quedó
    """
    if random==True:
        clasificacion,seeds=ejercicio2mod1(arreglo,nseeds,return_semillas=True)# ejecutamos el ejercicio uno y generamos la primera clasificacion y un arreglo con las semillas
        previous=seeds# centroide previo para la primera iteración el centroide previo son la semillas
    else:
        clasificacion=clasificacion_vertar3(arreglo,semillas)# genero clasificacion en base a mis centroides
        previous=semillas
    for iteracion in range (1,n_iteraciones+1):# iteraciones por default 100
        centroides=newcentroid(arreglo,clasificacion) # genero los centroides pormediando columnas
        clasificacion=clasificacion_vertar3(arreglo,centroides)# genero clasificacion en base a mis centroides
        distanciasC=norma(centroides,previous) # calculo la distancia entre un centroide y el centroide de la it anterior
        previous=centroides # actualizo el valor del centroide previo
        suma=np.sum(distanciasC)
        if suma<=tolerancia:
            break
    return clasificacion,centroides,iteracion



class complejo(): # llamé a mi clas complejo
    def __init__(self, real, imaginario): # el constructor inicializa los atributos de la clase
        self.real=real
        self.imaginario=imaginario
        return None
    def __add__(self, otro):# metodo de suma de números complejos con operador sobrecargado
        real=self.real + otro.real
        imaginario= self.imaginario + otro.imaginario        
        return complejo(real, imaginario) 
    def __sub__(self,otro): # metodo de resta de números complejos con operador sobrecargado
        real=self.real - otro.real
        imaginario= self.imaginario - otro.imaginario
        return complejo(real,imaginario)
    def modulo(self):
        modulo=np.sqrt((self.real**2)+(self.imaginario**2))
        return modulo
    def polar(self): #método para pasar a la forma polar
        r=self.modulo() # r es el módulo
        theta=np.degrees(np.arctan(self.imaginario/self.real)) # theta es el arcotangente de la division de la parte imaginaria entre la parte real
        # clasificación por cuadrantes
        if self.real>=0 and self.imaginario>=0: # si la parte real y la parte imaginaria son positivas 
            return (r,theta)
        elif self.real<0 and self.imaginario<0: #si la parte real es negativa y la parte imaginaria también
            theta=theta+180 
            return (r,theta)
        elif self.real<0 and self.imaginario>=0:  # si la parte real es negativa y la imaginaria positiva
            theta=180+theta
            return (r,theta)
        elif self.imaginario<0 and self.real>=0: #si la parte imaginaria es negativa y real positiva 
            theta=360+theta
            return (r,theta)
    def vectorial(r,theta): # forma vectorial
        theta=np.radians(theta)
        x=r*np.cos(theta) #parte real
        y=r*np.sin(theta) #parte imaginaria
        return complejo(x,y) #que devuelva un complejo formado con el resultado
    def __truediv__(self,otro): # método para dividir con operador sobrecargado
        numerador=self.polar()
        denominador=otro.polar()
        r=numerador[0]/denominador[0]
        theta=numerador[1]-denominador[1]
        return complejo.vectorial(r,theta)










