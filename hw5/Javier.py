#!/usr/bin/env python
# coding: utf-8

# In[7]:


# importamos las librerias que necesitaremos
import numpy as np
import matplotlib.pyplot as plt




#### Apertura y procesamiento inicial de datos 

def abrir(archivo):
    with open(archivo, mode='r') as datos: # abrimos el archivo
        contenido= datos.readlines() #contenido aqui es una lista de strings 
    return contenido
def procesado(contenido):
    ensayo=[] # declaro una lista vacía para guardar los registros correspondientes a un ensayo 
    string=contenido[0]# hay un str en contenido 
    #if string!='\n': # solo 60 str de contenido cumplen esta condición y esos son ensayos
    string=string.replace(','," ") # reemplazo las comas por espacios para poder usar split después
    string=string.split() #split hace cada número dentro del str de ensayos un str individual, y pone estos str dentro de una lista # ahora la variable string es una lista donde cada elemento es un str de un número flotante
    for numero in string: # itero cada elemento de la lista string
        numero=float(numero) # cada str de un número lo convierto a float 
        ensayo.append(numero)# cada float agrego con append a lista_n
    return ensayo
def importar(archivo): # función para importar datos
    content=abrir(archivo)
    Datos=procesado(content)
    return Datos


def bordesup(start,stop,paso,ventana,nv=False): # función para obtener los bordes superiores de mi ventana dadas las especificaciones de cada caso
    """
    Esta función cálcula los bordes superiores (para ventanas deterministicas) necesarios para calcular la tasa de disparo.
    alternativamente devuelve también el número de ventanas determinadas por estos bordes.
    La función recibe en donde empieza mi rango de tiempo y donde termina (en segundos), el tamaño del paso
    tamaño de la ventana. y cálcula los bordes superiores.alternativamente también cálcula 
    el número de ventanas determinadas por los bordes
    <parametros>
    start  es en donde empieza mi rango # en segundos
    stop es donde donde termina mi rango #segundos
    paso es el tamaño del paso #en segundos
    ventana es el tamaño de la ventana # en segundos
    nv es un parametro opcional que indica si regresa también el número de ventanas para el valor default es false
    <devuelve>
    np.array que contiene los valores de los bordes superiores (para  ventanas deterministicas)
    int que corresponde al número de ventanas determinadas por esos bordes
    """
    ventana,paso,stop,start=ventana*1000,paso*1000,stop*1000,start*1000
    sup=np.arange(start+ventana,stop+1,paso)/1000 #obtengamos los bordes superiores de mi ventana dadas las especificaciones
    if nv==True:
        nvent=(len(sup))
        return sup,nvent
    else:
        return sup



def tasa(ensayo,nventanas,paso,ventana):
    """
    Este algoritmo cálcula la tasa para un solo ensayo
    Esta función calcula la tasa de disparo con un algoritmo optimizado para reducir el uso de ciclos 
    for al mínimo.Para esto toma ventaja de la regularidad y el ordenamiento de los datos y hace uso de la 
    división entera y la vectorización.La función calcula la tasa de disparo dado un slice de una lista de
    una lista de numpy arrays donde cada array es un ensayo.
    <parámetros>
    TODOS LOS PARÁMETROS SON OBLIGATORIOS
    ensayos debe ser una lista de arrays o un slice de una lista de arrays donde cada array debe ser un ensayo.
    nventanas es el número de ventanas para el que estamos calculando
    paso es el tamaño de paso que vamos usar para calcular la tasa 
    ventana es el tamaño del la ventana que vamos a usar
    <devuelve>
    un numpy array donde cada elemento corresponde a la tasa de disparo para cada ventana determinística(a los pasos establecidos).
    """
    frecuencia=np.zeros(nventanas)# array vacío con un cero por cada ventana 
    ultima=ventana/paso # este número me dice en cuantas ventanas cae una espiga
    #############################
    vent=(ensayo)//paso    
    for nv in range(int(ultima)): # nv cuenta 0, 1,2,3,4 que es lo que necesito restar para ver si una espiga aparce en ventanas anteriores a la última en que aparece (((ensayo)//paso))
        ven=vent-nv  #los elementos de ((ensayo)//paso)-nv en la primera iteración son la última ventana en la que aparece un dato y en las siguientes iteraciones son las ventanas anteriores en las que puede aparecer
        ven=(ven[ven>=0])# la condicion [ven>=0] me evita que al restar nv aparezcan negativos cuando un valor no aparece en una ventana
        ven=(ven[ven<nventanas]).astype(int) # esto evita que si por ejemplo al tomar una espiga que cayo en los últimos pasos al hacer la división entera me de indices superiores a mi número de ventanas
        indices,cuentas=np.unique(ven,return_counts=True)#en este caso los valores unicos corresponden a los indices # y las cuentas a la cantidad de veces que aparece un índice
        frecuencia[indices]+=cuentas # matriz con las frecuencias por ventana
    tasa=(frecuencia/ventana) # calculamos la tasa de disparo
    return tasa
# tasa de disparo para un solo ensayo
def onerate(ensayo,nventanas,paso,ventana,start=0):
    """
    Este algoritmo cálcula la tasa para un solo ensayo
    Esta función calcula la tasa de disparo con un algoritmo optimizado para reducir el uso de ciclos 
    for al mínimo.Para esto toma ventaja de la regularidad y el ordenamiento de los datos y hace uso de la 
    división entera y la vectorización.La función calcula la tasa de disparo dado numpy array que es un ensayo.
    <parámetros>
    LOS PARÁMETROS OBLIGATORIOS SON:
    ensayos debe ser una lista de arrays o un slice de una lista de arrays donde cada array debe ser un ensayo.
    nventanas es el número de ventanas para el que estamos calculando
    paso es el tamaño de paso que vamos usar para calcular la tasa 
    ventana es el tamaño del la ventana que vamos a usar
    Parámetro opcional
    el parametro opcional
    start indica donde comienza el registro debido a que si empieza por debajo del tiempo cero para que funcione el algoritmo de
    división entera es necesario hacer una correción. si los datos comienzan por debajo de del tiempo cero y no se le especifica
    a la función en el parámetro start en que tiempo se empieza los datos calculados serán erroneos
    
    <devuelve>
    un numpy array donde cada elemento corresponde a la tasa de disparo para cada ventana determinística(a los pasos establecidos).
    """
    if (start>0): # el parametro opcional por default vale cero
        raise NameError('El parametro start debe ser un valor negativo o cero')
    else:
        frecuencia=np.zeros(nventanas)# array vacío con un cero por cada ventana 
        ultima=ventana/paso # este número me dice en cuantas ventanas cae una espiga
        vent=((ensayo-start)//paso)# hay que sumar un -start si empezamos el resgistro de datos en  un punto distinto de cero para hacer el ajuste del algoritmo al caso #los elementos de la matriz ((ensayo-start)//paso) son la última ventana en que aparece un dato
        for nv in range(int(ultima)): # nv cuenta 0, 1,2,3,4... que es lo que necesito restar para ver si una espiga aparce en ventanas anteriores a la última en que aparece (((ensayo-start)//paso))
            ven=vent-nv #los elementos de ((ensayo-start)//paso)-nv en la primera iteración son la última ventana en la que aparece un dato y en las siguientes iteraciones son las ventanas anteriores en las que puede aparecer
            ven=(ven[ven>=0])# la condicion [ven>=0] me evita que al restar nv aparezcan negativos cuando un valor no aparece en una ventana
            ven=(ven[ven<nventanas]).astype(int) # esto evita que si por ejemplo al tomar una espiga que cayo en los últimos pasos al hacer la división entera me de indices superiores a mi número de ventanas
            indices,cuentas=np.unique(ven,return_counts=True)#en este caso los valores unicos corresponden a los indices # y las cuentas a la cantidad de veces que aparece un índice
            frecuencia[indices]+=cuentas # matriz con las frecuencias por ventana
    tasa=(frecuencia/ventana) # calculamos la tasa de disparo
    return tasa





