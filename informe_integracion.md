# Informe de Integración — Prototipo Descenso de Gradiente

**Autor:** Felipe Jiménez
**Proyecto:** Optimización de algoritmos de Inteligencia Artificial mediante Cálculo Diferencial
**Repositorio:** `prototipo-descenso-de-gradiente`

## Introducción

Este informe documenta, tarea por tarea, el desarrollo del notebook `descenso_gradiente.ipynb`, siguiendo el roadmap de 6 etapas definido en el README: generación del dataset, función de costo y derivada, descenso de gradiente, búsqueda aleatoria, métricas y gráficos. Por cada tarea explico qué implementé, cómo lo hice, por qué tomé cada decisión y cómo verifiqué que funcionara correctamente.

---

## Tarea 1: Generación del dataset

### Qué hice

Implementé la celda 2 del notebook para generar un dataset sintético de clasificación binaria con `sklearn.datasets.make_classification`, y lo exporté a `dataset.csv` con las columnas `x1`, `x2`, `y`.

### Cómo lo hice

Usé los siguientes parámetros:

- `n_samples=500`
- `n_features=2`
- `n_informative=2`
- `n_redundant=0`
- `n_clusters_per_class=1`
- `class_sep=1.5`
- `random_state=42`

El resultado lo volqué a un `DataFrame` de pandas y lo exporté con `to_csv("dataset.csv", index=False)`.

### Por qué lo hice así

- Elegí 500 muestras porque es suficiente para que la curva de convergencia del descenso de gradiente se vea suave, sin hacer las iteraciones más lentas de lo necesario.
- Usé solo 2 features (`x1`, `x2`) a propósito, porque así puedo graficar la frontera de decisión en 2D en la tarea 6, que es justo lo que necesitamos para la diapositiva 12 del PPT.
- Puse `n_informative=2` y `n_redundant=0` para que ambas variables aporten señal real y no haya ruido innecesario.
- Fijé `n_clusters_per_class=1` para que las clases queden razonablemente separables y el problema sea fácil de explicar visualmente.
- Elegí `class_sep=1.5` como un punto intermedio: las clases se distinguen con claridad, pero el problema no es trivial, así que sigue habiendo error que el modelo tiene que minimizar.
- Fijé `random_state=42` para que el dataset sea reproducible: cualquiera del equipo que corra la celda obtiene exactamente los mismos 500 puntos.
- Decidí mantener `dataset.csv` versionado en git (sin agregarlo a `.gitignore`) para que todo el equipo trabaje sobre el mismo dataset, sin depender de que cada uno lo regenere por su cuenta.

### Cómo lo verifiqué

Ejecuté el código de forma independiente fuera del notebook y confirmé que genera 500 muestras con 2 features sin errores, y que `dataset.csv` se crea correctamente con las columnas esperadas.

---

## Tarea 2: Función de costo (MSE) y su derivada

### Qué hice

Implementé en la celda 4 el modelo de predicción, la función de costo y su gradiente de forma explícita (sin autograd): `sigmoid`, `predict`, `cost_function` y `gradient`.

### Cómo lo hice

Definí el modelo como una combinación lineal seguida de una sigmoide:

```
z = X·W + b
ŷ = σ(z) = 1 / (1 + e^(-z))
```

Con `b` separado de `W` (no como una columna extra de 1s en `X`), porque así queda más claro mostrar la derivada por separado en el informe.

La función de costo es el Error Cuadrático Medio:

```
J(W, b) = (1/n) · Σ (ŷᵢ − yᵢ)²
```

Y derivé el gradiente aplicando la regla de la cadena paso a paso:

```
dJ/dŷ = (2/n)·(ŷ − y)
dŷ/dz = ŷ·(1 − ŷ)          (derivada de la sigmoide)
dz/dW = x  ;  dz/db = 1

dJ/dW = (2/n) · Σ (ŷᵢ − yᵢ) · ŷᵢ·(1−ŷᵢ) · xᵢ
dJ/db = (2/n) · Σ (ŷᵢ − yᵢ) · ŷᵢ·(1−ŷᵢ)
```

Cada término de esa descomposición quedó reflejado directamente en el código, para que la fórmula matemática y la implementación coincidan 1:1.

### Por qué lo hice así

- Usé MSE en lugar de log-loss/entropía cruzada (que sería lo "estándar" en clasificación) porque el objetivo del curso es Cálculo Diferencial: quiero mostrar la derivada paso a paso con regla de la cadena usando una función de costo que ya conocemos de regresión, tal como lo pide el README.
- Implementé el gradiente derivando a mano cada término (`dJ/dŷ`, `dŷ/dz`, `dz/dW`) en vez de usar una librería de diferenciación automática, porque ese es justamente el punto central del proyecto: aplicar cálculo diferencial explícito, no delegarlo en una caja negra.
- Dejé el factor `2/n` explícito en vez de absorberlo en la tasa de aprendizaje `α`, para que la fórmula matemática documentada y el código sean exactamente equivalentes y fáciles de revisar.

### Cómo lo verifiqué

No me bastaba con que el código corriera sin errores; necesitaba confirmar que la derivada estuviera matemáticamente bien calculada. Para eso comparé el gradiente **analítico** (la fórmula que programé) contra el gradiente **numérico**, calculado por diferencias finitas:

```
dJ/dW ≈ (J(W+ε) − J(W−ε)) / (2ε)
```

con `ε = 1e-6`. Los dos coincidieron hasta la sexta cifra decimal (`dW analítico: [0.0676, -0.1646]` vs `dW numérico: [0.0676, -0.1646]`), lo cual es el chequeo estándar para detectar errores de signo o factores faltantes en la regla de la cadena. También corrí una prueba con `W=0, b=0`, donde el costo inicial dio 0.25 — justo lo esperado, porque con pesos en cero `ŷ=0.5` para todos los puntos, y `(0.5−1)²` o `(0.5−0)²` promedian 0.25.

---

## Tarea 3: Descenso de Gradiente

### Qué hice

Implementé en la celda 6 la función `gradient_descent`, que aplica el bucle iterativo completo usando `cost_function` y `gradient` de la tarea 2, guardando el historial de costo en cada paso hasta que converge.

```python
def gradient_descent(X, y, alpha=ALPHA, max_iter=MAX_ITER, tol=TOL):
    W = np.zeros(X.shape[1])
    b = 0.0
    cost_history = [cost_function(X, y, W, b)]

    for i in range(1, max_iter + 1):
        dW, db = gradient(X, y, W, b)
        W = W - alpha * dW
        b = b - alpha * db
        cost_history.append(cost_function(X, y, W, b))

        if abs(cost_history[-2] - cost_history[-1]) < tol:
            break

    return W, b, cost_history
```

Después la llamo, calculo la precisión final y muestro los resultados.

### Para qué (propósito dentro del proyecto)

Esta es la pieza central del prototipo: toma el gradiente calculado en la tarea 2 y lo usa para mover los parámetros iterativamente hacia el mínimo del error. El `cost_history` que devuelve es lo que la tarea 6 va a graficar como curva de convergencia, y `W_gd`, `b_gd`, `iters_gd` son lo que la tarea 5 va a comparar contra la Búsqueda Aleatoria.

### Cómo lo hice

El bucle sigue la regla de actualización:

```
W(t+1) = W(t) − α · dJ/dW
b(t+1) = b(t) − α · dJ/db
```

Parto de `W = 0`, `b = 0`, y en cada iteración calculo el gradiente, actualizo los parámetros y guardo el nuevo costo en `cost_history`. Corto el bucle cuando el costo deja de mejorar más que una tolerancia (`|J(t) − J(t-1)| < tol`) o al llegar a un máximo de iteraciones, lo que ocurra primero. Al final calculo también la precisión, clasificando como 1 cuando `ŷ ≥ 0.5`.

Los hiperparámetros que dejé son:

- `ALPHA = 1.0` (tasa de aprendizaje)
- `MAX_ITER = 5000` (tope de iteraciones)
- `TOL = 1e-6` (umbral de convergencia)

### Por qué lo hice así

| Decisión | Cómo | Por qué |
|---|---|---|
| Regla de actualización | `W ← W − α·dW`, `b ← b − α·db`, aplicada en cada iteración | Es la definición estándar de descenso de gradiente: moverse en la dirección opuesta al gradiente (que apunta hacia donde crece el costo) |
| Inicialización en cero | `W = np.zeros(...)`, `b = 0.0` | Punto de partida neutro y reproducible; con esto el costo inicial es 0.25, coincide con lo verificado en la tarea 2 |
| `α = 1.0` | Elegido probando 0.05, 0.1, 0.5, 1.0, 2.0 | Con tasas bajas (0.05–0.5) el modelo converge muy lento y no baja lo suficiente en un número razonable de iteraciones; con 1.0 converge rápido sin oscilar ni divergir; 2.0 no mejora lo suficiente como para justificar el riesgo de una tasa más agresiva |
| `tol = 1e-6` | Corta el bucle cuando `\|J(t)−J(t-1)\| < tol` | Con MSE + sigmoide el gradiente se va achicando cerca de los extremos (por el factor `ŷ(1-ŷ)`, que tiende a 0 cuando `ŷ→0` o `ŷ→1`), así que una tolerancia más estricta (`1e-9`) nunca "cierra" el bucle aunque ya casi no haya mejora real |
| `max_iter = 5000` | Tope duro del `for` | Red de seguridad para que el bucle no corra indefinidamente si no converge; en la práctica el criterio de tolerancia corta mucho antes |
| Guardar `cost_history` completo | Se agrega un valor a la lista en cada iteración, no solo el final | La tarea 6 necesita la serie completa para graficar la curva de convergencia, no solo el resultado final |

### Cómo lo verifiqué

Corrí el bucle de forma independiente fuera del notebook y comprobé tres cosas concretas:

1. **El costo baja** de 0.25 (inicial, con `W=0,b=0`) a 0.0294 (final).
2. **Es monótonamente decreciente** en cada iteración — lo comprobé con un `assert` que revisa que `cost_history[i] >= cost_history[i+1]` para todo `i`. Esto es justamente lo que se espera de un descenso de gradiente bien configurado (con una `α` demasiado alta, el costo oscilaría o subiría en algún punto).
3. **Converge antes del límite**: se detiene en 2445 iteraciones (de un tope de 5000), con precisión final de 96.4% sobre el propio dataset.

---

## Próximos pasos

- **Tarea 4:** implementar la Búsqueda Aleatoria como baseline de comparación.
- **Tarea 5:** medir CPU, RAM y tiempo real con `psutil`, además de iteraciones hasta converger y precisión final, para ambos métodos.
- **Tarea 6:** graficar la curva de error vs. iteración y la comparación entre métodos, exportando los resultados como PNG.
