# 🏠 Predicción de consumo eléctrico en función del tamaño de la vivienda
## 🎯 Descripción del proyecto
Proyecto que predice el consumo eléctrico en función al tamaño de la vivienda usando regresión lineal.
## ❓ Contexto del problema
El objetivo es realizar regresión lineal con una sola variable del conjunto de datos: el tamaño de la vivienda. Hemos escogido solo una variable por simplicidad para crear un modelo sencillo y porque se pretendía programar de forma manual el modelo de regresión lineal, sin paquetes externos. Resulta por tanto un proyecto interesante para familizarse con conceptos básicos de ML y aprender a implementar un modelo de aprendizaje supervisado. En futuros proyectos se implementará un modelo más complejo usando regresión lineal múltiple.
## 📃 Dataset utilizado
Este data set se corresponde con "Energy Consumption Dataset - Linear Regression" (https://www.kaggle.com/datasets/govindaramsriram/energy-consumption-dataset-linear-regression?resource=download).

*Descripción del dataset*:

This dataset is designed for predicting energy consumption based on various building features and environmental factors. It contains data for multiple building types, square footage, the number of occupants, appliances used, average temperature, and the day of the week. The goal is to build a predictive model to estimate energy consumption using these attributes.

The dataset can be used for training machine learning models such as linear regression to forecast energy needs based on the building's characteristics. This is useful for understanding energy demand patterns and optimizing energy consumption in different building types and environmental conditions.
## 🛠️ Metodología
### 1. Lectura de los datos
```bash
# Leemos los datos
df = pd.read_csv('train_energy_data.csv')
df
```
### 2. Visualización de los datos
```bash
# Graficamos
X = df['Square Footage']
Y = df['Energy Consumption']

plt.scatter(X, Y, s=20, c='blue')
plt.xlabel('Square Footage (ft^2)')
plt.ylabel('E (kWh)')
plt.grid()
```
### 3. Aplicación del descenso del gradiente
- La recta que buscaremos tendrá la siguiente forma:
```math
$$\hat{y}=wx+b,$$
```
donde $\hat{y}$ se corresponde con el valor predicho del consumo en base a un valor de tamaño de la vivienda, y $w$ y $b$ serán los parámetros a determinar.

- Para hallar los mejores $w$ y $b$ que se ajusten a la nube de puntos, construimos la función de coste: 
```math
$$J(w,b)=1/2m \sum_{i=1}^{m} \left(\hat{y}_i-y_i\right)^2 = 1/2m \sum_{i=1}^{m} \left(wx_i+b-y_i\right)^2,$$
```
siendo $m$ el número total de elementos del conjunto de entrenamiento y $y_i$ cada uno de los valores reales para cada tamaño de vivienda $x_i$.

- Lo único que tenemos que hacer ahora es minimizar la función de coste, esto es, encontrar aquellos $w$ y $b$ que hagan mínima la función de coste. Con ellos, tendremos la mejor recta posible que se ajuste a los datos. Para esta tarea de optimización, usaremos el siguiente conocido algoritmo: el descenso del gradiente. Matemáticamente:
```math
$$\vec{v}_{k+1}=\vec{v}_k-\alpha \vec{\nabla} J(w,b),$$
```

siendo $\vec{v}$ un par $(w,b)$ y $\alpha$ el ratio de aprendizaje (learning rate). Por tanto, $\vec{v}_{0}$ será el valor de $w$ y $b$ que escojamos inicialmente y que dé comienzo el método (la iteración inicial).
## 📈 Visualización de los resultados clave
- **Consumo base**: $2878.04\, kWh$
- **Consumo adicional por pie al cuadrado**: $0.05\,kWh/ft^2$
![Consumo eléctrico (kWh) frente a tamaño vivienda (ft^2)](/regresionlineal (2)_imagen.png)
## 💼 Conclusiones clave
- Existe un **consumo base garantizado** de ~2878 kWh por vivienda, independientemente de su tamaño.  
- Por cada pie cuadrado adicional de superficie, el consumo eléctrico promedio aumenta en **0.05 kWh**.  

**Implicaciones de negocio:**
- **Ingresos mínimos estables por cliente**, ya que todo hogar tiene un consumo fijo.  
- **Mayor potencial de consumo en viviendas grandes**, lo que permite segmentar clientes y diseñar ofertas específicas.  
- **Mejor planificación de la demanda**, al estimar consumo total a partir de la distribución de tamaños de vivienda.  

