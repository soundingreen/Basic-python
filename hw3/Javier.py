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
