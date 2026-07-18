# Prototipo: Descenso de Gradiente

Proyecto de la asignatura Cálculo Diferencial — Optimización de algoritmos de Inteligencia Artificial: minimizando el error mediante el Cálculo Diferencial.

**Integrantes:** Raúl Guzmán, Felipe Jiménez, Fernando Lagos, Marcelo Hermosilla
**Docente:** Silvia Elizabeth Flores Díaz

## Objetivo

Implementar desde cero el algoritmo de Descenso de Gradiente en Python (NumPy), usando el cálculo de derivadas para minimizar la función de costo (Error Cuadrático Medio) de un modelo de clasificación, y comparar su eficiencia contra una búsqueda aleatoria de parámetros.

## Estructura del notebook (`descenso_gradiente.ipynb`)

1. Generación del dataset — datos sintéticos de clasificación binaria (`sklearn.datasets.make_classification`), exportados a `dataset.csv`.
2. Función de costo (MSE) y su derivada — implementación explícita de ∂J/∂w.
3. Descenso de Gradiente — bucle iterativo W ← W − α·∂J/∂w.
4. Búsqueda Aleatoria — baseline de comparación.
5. Métricas — CPU, RAM y tiempo real (medidos con `psutil`), iteraciones hasta converger, precisión final.
6. Gráficos — curva de error vs. iteración y comparación entre métodos, exportados como PNG.

## Cómo ejecutar

```bash
pip install -r requirements.txt
jupyter notebook descenso_gradiente.ipynb
```

## Contexto del proyecto

Este repositorio es el producto final de la Etapa 3 (Presentación del producto) del proyecto ABPro. La presentación completa (PPT, bitácora, cronograma, evaluaciones) está en la carpeta `Documentos/` del entregable del curso, no en este repositorio.
