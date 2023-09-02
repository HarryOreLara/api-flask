import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



###Declaramos las entradas

#pulgadas
entrada = np.array([1,6,30,7,10,70,345,600,1004,43], dtype=float)

#metros
resultados = np.array([0.0254, 0.1524, 0.762, 0.1778, 0.254,1.778,8.763,15.24,25.5016,1.0922], dtype=float)


#---Topografia de la red
capa1 = tf.keras.layers.Dense(units=1, input_shape = [1])

modelo = tf.keras.Sequential([capa1])


#------Asignamos optimizaros y metrica de perdida

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)


print("Entrenando la red")


##Entrnamos el modelo

entrenamieto = modelo.fit(entrada, resultados, epochs=500, verbose=True)


##gUARDAR LA RED NEURONAL
modelo.save('RedNeuronal.h5')
modelo.save_weights('peros.h5')



##Observar el comportamiento de la red
plt.xlabel('Ciclos de entrenamiento')
plt.ylabel('Errores')
plt.plot(entrenamieto.history["loss"])
plt.show()





##Verificar que la red se entreno
print("Terminamos")

##prediccion
i = input("Ingresar el valor en pulgadas")
i = float(i)
prediccion = modelo.predict([i])
print('El valor en metros es: ', str(prediccion))