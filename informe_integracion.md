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

## Tarea 4: Búsqueda Aleatoria (baseline)

### Qué hice

Implementé en la celda 8 la función `random_search`: en cada intento sortea `W` y `b` al azar (uniforme) dentro de un rango, evalúa `cost_function`, y se queda con el mejor par encontrado hasta el momento.

```python
def random_search(X, y, max_iter=MAX_ITER_RS, patience=PATIENCE,
                   w_range=W_RANGE, b_range=B_RANGE, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    best_W = rng.uniform(*w_range, size=X.shape[1])
    best_b = rng.uniform(*b_range)
    best_cost = cost_function(X, y, best_W, best_b)
    cost_history = [best_cost]
    no_improve = 0

    for i in range(1, max_iter + 1):
        W_candidate = rng.uniform(*w_range, size=X.shape[1])
        b_candidate = rng.uniform(*b_range)
        cost_candidate = cost_function(X, y, W_candidate, b_candidate)

        if cost_candidate < best_cost:
            best_W, best_b, best_cost = W_candidate, b_candidate, cost_candidate
            no_improve = 0
        else:
            no_improve += 1

        cost_history.append(best_cost)

        if no_improve >= patience:
            break

    return best_W, best_b, cost_history
```

### Para qué (propósito dentro del proyecto)

Es el baseline "sin cálculo" contra el que se compara el Descenso de Gradiente: mismo modelo (`predict`, `cost_function`), mismo dataset, pero sin usar la derivada para decidir hacia dónde moverse — solo prueba puntos al azar y se queda con el mejor. La tarea 5 va a comparar `iters_rs` vs `iters_gd`, tiempo, CPU/RAM y precisión final entre ambos métodos.

### Cómo lo hice

- Rango de búsqueda: `W_RANGE = B_RANGE = (-10, 10)`.
- Presupuesto máximo: `MAX_ITER_RS = 5000` (mismo tope que el Descenso de Gradiente).
- Corte anticipado: si el mejor costo no mejora en `PATIENCE = 500` intentos seguidos, se detiene.
- Guardo en `cost_history` el **mejor costo hasta ese punto** en cada iteración (no el costo del intento actual), para que la curva sea comparable con la del Descenso de Gradiente en la tarea 6.

### Por qué lo hice así

| Decisión | Cómo | Por qué |
|---|---|---|
| `max_iter = 5000` | Mismo tope que `gradient_descent` | Para que la comparación de la tarea 5 sea justa: ambos métodos parten del mismo presupuesto máximo de intentos |
| Rango `(-10, 10)` | `rng.uniform(*w_range, ...)` para `W` y `b` | Cubre cómodamente la solución que encontró el Descenso de Gradiente (`W≈[-2.1, 5.6], b≈-2.0`), sin ser tan amplio como para que la búsqueda sea insensata |
| `patience = 500` | Se detiene si `no_improve >= patience` | A diferencia del Descenso de Gradiente, que tiene un criterio de convergencia natural basado en el gradiente, la búsqueda aleatoria no "sabe" cuándo parar por sí sola; sin este corte correría siempre hasta `max_iter` aunque ya no esté mejorando, distorsionando la comparación de tiempo/iteraciones de la tarea 5 |
| `seed = RANDOM_STATE` | Misma semilla 42 que el resto del notebook | Reproducibilidad: el resultado es el mismo cada vez que se corre |
| Guardar el mejor costo por iteración, no el del intento actual | `cost_history.append(best_cost)` dentro del bucle | Así la curva es monótonamente no creciente y comparable visualmente con la del Descenso de Gradiente en la tarea 6 |

### Cómo lo verifiqué

Corrí la celda de forma independiente y comprobé con un `assert` que el "mejor costo" es no creciente en cada iteración (por construcción, nunca puede empeorar). Resultado: converge en 556 iteraciones (bastante antes del tope de 5000), con costo final 0.0276 y precisión 96.8% — muy cercano a lo que logró el Descenso de Gradiente (0.0294 de costo, 96.4% de precisión), pero llegando ahí en 556 intentos en vez de 2445. Tiene sentido: con solo 3 parámetros (`W1, W2, b`), el muestreo aleatorio puede ser sorprendentemente competitivo en cantidad de intentos. La comparación real de eficiencia (tiempo de cómputo, CPU, RAM) se hará en la tarea 5, que es donde se espera que la diferencia entre ambos métodos se note con más claridad.

---

## Tarea 5: Métricas

### Qué hice

Implementé en la celda 10 la función `measure`, que envuelve la ejecución de `gradient_descent` y `random_search` para capturar tiempo real, CPU y RAM con `psutil`, además de reutilizar las iteraciones y precisión que ya calculan ambas funciones. Junté todo en una tabla `metricas` (`pandas.DataFrame`) y la exporté a `metricas.csv`.

```python
def measure(func, *args, **kwargs):
    process = psutil.Process()
    process.cpu_percent(interval=None)  # calentamiento
    mem_before = process.memory_info().rss

    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    t1 = time.perf_counter()

    cpu_pct = process.cpu_percent(interval=None)
    mem_after = process.memory_info().rss

    elapsed = t1 - t0
    ram_delta_mb = (mem_after - mem_before) / (1024 ** 2)
    return result, elapsed, cpu_pct, ram_delta_mb
```

### Para qué (propósito dentro del proyecto)

Es la comparación cuantitativa entre el método basado en cálculo (Descenso de Gradiente) y el baseline sin cálculo (Búsqueda Aleatoria), que es justamente el objetivo del prototipo según el README. `metricas.csv` queda listo para pegarse directo en la diapositiva del PPT, y la tabla es la que la tarea 6 va a usar para graficar la comparación entre métodos.

### Cómo lo hice

Volví a ejecutar ambos métodos (no reutilicé los resultados de las celdas 6 y 8) porque esas celdas no habían medido CPU/RAM/tiempo, solo costo e iteraciones. `measure` toma cualquier función y sus argumentos, mide el tiempo con `time.perf_counter()`, y usa `psutil.Process()` para leer CPU y memoria (RSS) antes y después de la llamada. Con los resultados arme una tabla con columnas: Método, Iteraciones, Tiempo (s), CPU (%), RAM (MB), Costo final, Precisión final — una fila por método — y la exporté con `metricas.to_csv("metricas.csv", index=False)`.

### Por qué lo hice así

| Decisión | Cómo | Por qué |
|---|---|---|
| Volver a correr ambos métodos en vez de reusar resultados previos | Llamo `gradient_descent(X, y)` y `random_search(X, y)` de nuevo dentro de `measure` | Las tareas 3 y 4 no midieron CPU/RAM/tiempo; para obtener esas métricas hay que perfilar la ejecución real, no se pueden calcular después del hecho |
| "Calentar" `cpu_percent()` con una llamada previa | `process.cpu_percent(interval=None)` antes de medir | La primera lectura de `cpu_percent()` siempre devuelve 0.0 porque no tiene un punto de referencia anterior; sin este paso el CPU% medido saldría mal para ambos métodos |
| Medir RAM como diferencia (`mem_after - mem_before`), no como valor absoluto | `process.memory_info().rss` antes y después | El RSS absoluto incluye toda la memoria del proceso de Python (intérprete, librerías cargadas, etc.), no solo lo que usó la función; la diferencia aísla el consumo atribuible a esa llamada |
| Función genérica `measure(func, *args, **kwargs)` en vez de duplicar la instrumentación para cada método | Se llama una vez con `gradient_descent` y otra con `random_search` | Evita repetir el mismo código de medición dos veces; ambas funciones devuelven la misma forma de resultado `(W, b, cost_history)`, así que la misma envoltura sirve para las dos |
| Exportar a `metricas.csv` además de mostrar la tabla en el notebook | `metricas.to_csv("metricas.csv", index=False)` | Lo pediste explícitamente, para poder usar los números directo en la diapositiva del PPT sin tener que copiarlos a mano desde el notebook |

### Cómo lo verifiqué

Corrí la celda de forma independiente y generé el `metricas.csv` real del proyecto. Los resultados obtenidos:

| Método | Iteraciones | Tiempo (s) | CPU (%) | RAM (MB) | Costo final | Precisión final |
|---|---|---|---|---|---|---|
| Descenso de Gradiente | 2445 | 0.0804 | 97.1 | 0.156 | 0.0294 | 96.4% |
| Búsqueda Aleatoria | 556 | 0.0103 | 0.0 | 0.000 | 0.0276 | 96.8% |

Estos números confirman lo que ya habíamos visto en las tareas 3 y 4 (iteraciones y precisión), y agregan la parte nueva: en tiempo real, la Búsqueda Aleatoria es ~8 veces más rápida que el Descenso de Gradiente en este problema de solo 3 parámetros — consistente con que necesita casi 5 veces menos iteraciones. El CPU (%) y RAM (MB) de la Búsqueda Aleatoria salieron en 0 porque su ejecución es tan corta (10 ms) que `psutil` no alcanza a registrar un delta medible en esa ventana de tiempo; no es un error, es una limitación esperable de medir procesos muy cortos con esta herramienta, y vale la pena mencionarlo así en el informe final si se usa este dato en el PPT.

---

## Tarea 6: Gráficos

### Qué hice

Implementé en la celda 12 tres gráficos, cada uno exportado a PNG:

1. **`curva_convergencia.png`** — costo (MSE) vs. iteración, con las dos curvas (Descenso de Gradiente y Búsqueda Aleatoria) superpuestas en el mismo eje.
2. **`comparacion_metodos.png`** — tres gráficos de barras lado a lado (iteraciones, tiempo, precisión final), usando directamente la tabla `metricas` de la tarea 5.
3. **`frontera_decision.png`** — la frontera de decisión del modelo entrenado con Descenso de Gradiente, dibujada sobre el dataset original (`x1` vs `x2`, coloreado por clase).

### Para qué (propósito dentro del proyecto)

Son los gráficos que van directo a la diapositiva 12 del PPT (según indica el README). La curva de convergencia y la comparación de métricas son las que pide explícitamente la tarea 6; agregué la frontera de decisión porque fue justamente la razón por la que en la tarea 1 elegí generar el dataset con solo 2 features — quería poder mostrar visualmente qué aprendió el modelo, no solo los números.

### Cómo lo hice

- Gráfico 1: `ax1.plot(cost_history_gd, ...)` y `ax1.plot(cost_history_rs, ...)` en el mismo `Axes`, cada serie con su propio eje X implícito (el índice de la lista), así que las curvas tienen distinto largo (2445 vs. 556 puntos) pero se comparan igual de forma visual.
- Gráfico 2: `plt.subplots(1, 3, ...)` con un subplot de barras por métrica, tomando las columnas directamente del DataFrame `metricas` que ya se había armado en la tarea 5 (no dupliqué números a mano).
- Gráfico 3: generé una grilla de puntos (`np.meshgrid`) sobre el rango de `x1` y `x2`, evalué `predict(grid, W_gd, b_gd)` en cada punto de la grilla, y usé `contourf`/`contour` para pintar la región donde el modelo predice cada clase (con el corte en 0.5), superponiendo el `scatter` de los datos reales.
- Cada figura se exporta con `fig.savefig("...", dpi=150)` antes de `plt.show()`.

### Por qué lo hice así

| Decisión | Cómo | Por qué |
|---|---|---|
| Curvas superpuestas en un solo gráfico, no en subplots separados | `ax1.plot(...)` dos veces sobre el mismo `Axes` | Así se ve directamente cuál método converge más rápido y a qué costo final, que es el punto central de la comparación |
| Reutilizar la tabla `metricas` de la tarea 5 para el gráfico 2 | `axes[i].bar(metodos, metricas["..."])` | Evita transcribir números a mano (fuente de errores); si se cambian los hiperparámetros de las tareas 3/4, el gráfico se actualiza solo al re-ejecutar el notebook |
| `dpi=150` en las exportaciones | Parámetro de `savefig` | Buena resolución para verse nítido en una diapositiva de PPT sin generar archivos pesados |
| Agregar la frontera de decisión aunque no la pedía literalmente el enunciado de la tarea 6 | Grilla + `predict` + `contourf` | Cumple la promesa que me hice en la tarea 1 al elegir 2 features: mostrar qué aprendió el modelo, no solo la curva de error — es el gráfico más intuitivo para explicarle a alguien que no vio el código |
| Usar el modelo de la tarea 3 (`W_gd`, `b_gd`), no el re-medido en la tarea 5 (`W_gd_m`) | Referencio las variables originales de la celda 6 | Son numéricamente idénticos (Descenso de Gradiente es determinista, parte siempre de `W=0,b=0`), pero uso las variables "canónicas" del entrenamiento por trazabilidad |

### Cómo lo verifiqué

Corrí las tres celdas de forma independiente, confirmé que los 3 PNG se generan (39 KB, 60 KB y 128 KB respectivamente) y revisé cada imagen:

- La curva de convergencia muestra claramente que la Búsqueda Aleatoria baja más brusco al principio (por el patrón de "mejor hasta ahora") y se estabiliza antes, mientras el Descenso de Gradiente desciende más suave y sigue afinando el costo por más iteraciones.
- El gráfico de barras confirma visualmente los números de la tarea 5: la Búsqueda Aleatoria gana en iteraciones y tiempo, y ambos métodos quedan prácticamente empatados en precisión final.
- La frontera de decisión separa razonablemente bien las dos clases (solo un puñado de puntos quedan del lado incorrecto), coherente con la precisión de 96.4% medida en la tarea 3.

### Corrección: paleta de colores invertida en la frontera de decisión

En la primera versión del gráfico 3, el fondo sombreado usaba colores fijos (`colors=["#fde0dd", "#deebf7"]`, rosado para la zona de baja probabilidad y celeste para la de alta) elegidos sin fijarme en la convención de `coolwarm`, el colormap que uso para los puntos (`cmap="coolwarm"`, donde azul = clase 0 y rojo = clase 1). El resultado visual quedaba con los puntos azules cayendo en la zona rosada y los puntos rojos en la zona celeste — a simple vista parece que el modelo clasifica al revés.

Verifiqué numéricamente que **el modelo no tiene ningún error**: los puntos de clase 1 tienen probabilidad predicha alta (0.64–0.95) y los de clase 0 probabilidad baja (0.00–0.45), exactamente como debe ser. El problema era solo visual: dos paletas de colores con la convención invertida entre sí. Corregí intercambiando los colores del fondo (`colors=["#deebf7", "#fde0dd"]`, celeste para la zona de clase 0 y rosado para la zona de clase 1) para que coincidan con `coolwarm`. Con el fix, el gráfico se lee de forma intuitiva: puntos rojos en zona rosada, puntos azules en zona celeste, con las mismas pocas excepciones (errores reales del modelo) que ya sabíamos que existían por la precisión de 96.4%.

**Lección para el informe/defensa:** al graficar predicciones sobre datos reales, siempre hay que verificar que la paleta del fondo (regiones de predicción) y la paleta de los puntos (etiquetas reales) usen la misma convención de color — si no, un modelo perfecto puede parecer roto solo por el gráfico.

### Cómo interpretar cada gráfico

**`curva_convergencia.png` — Costo vs. Iteración**

- Eje X: número de iteración. Eje Y: costo (MSE) en esa iteración.
- Las dos curvas bajan porque ambos métodos buscan minimizar el error, pero la **forma** de la caída es lo que importa:
  - *Descenso de Gradiente (azul):* desciende de forma suave y continua. Es lo que predice el cálculo: en cada paso se mueve exactamente en la dirección que más reduce el costo (`-α·∂J/∂W`), así que nunca "retrocede" ni da saltos — es monótona y progresiva.
  - *Búsqueda Aleatoria (naranja):* baja en **escalones**. `cost_history` guarda el mejor costo encontrado *hasta el momento*, así que la curva queda plana mientras prueba puntos al azar que no mejoran, y cae de golpe cuando por suerte encuentra un punto mejor. No tiene noción de "dirección".
- Lectura para la presentación: el Descenso de Gradiente converge de forma **informada** (usa la derivada); la Búsqueda Aleatoria converge de forma **ciega** (prueba y error). Esa es la diferencia conceptual central del proyecto.

**`comparacion_metodos.png` — Iteraciones, Tiempo, Precisión**

- Barra más baja = mejor en Iteraciones y Tiempo (más eficiente). Barra más alta = mejor en Precisión.
- En este dataset, la Búsqueda Aleatoria "gana" en iteraciones y tiempo, y ambos métodos quedan casi empatados en precisión (~96–97%).
- **Punto importante para la defensa:** esto puede sonar contraintuitivo si el argumento del proyecto es que el cálculo es mejor. La explicación es que el problema tiene solo 3 parámetros (`W1, W2, b`) — un espacio tan chico que probar puntos al azar es viable. Con muchos más parámetros (una red neuronal real, con miles o millones de pesos), la Búsqueda Aleatoria se vuelve inviable exponencialmente, mientras que el Descenso de Gradiente sigue escalando porque siempre sabe hacia dónde moverse. Ese contraste (funciona en este caso chico, pero no escala) es el argumento real a favor del cálculo diferencial, y conviene explicitarlo al mostrar este gráfico.

**`frontera_decision.png` — Frontera de decisión sobre el dataset**

- Cada punto es una muestra: rojo/azul según su clase real (columna `y`, `cmap="coolwarm"`).
- Las regiones de fondo son lo que el modelo predice para cada zona del plano `x1`-`x2` (celeste = predicción clase 0, rosado = predicción clase 1, alineado con los colores de los puntos tras la corrección de arriba).
- La línea negra es la frontera de decisión: donde `ŷ = σ(X·W+b) = 0.5`, el punto exacto donde el modelo "duda" entre las dos clases. Es una línea **recta** porque el modelo es lineal (`z = X·W+b`) antes de pasar por la sigmoide — la sigmoide curva la probabilidad, no la forma de la frontera.
- Cómo leerlo: los puntos del lado de su color correcto están bien clasificados; los pocos que caen del lado opuesto son los errores que explican por qué la precisión es 96.4% y no 100%.
- Esta línea está determinada directamente por `W_gd` y `b_gd` — es la traducción visual concreta de "para qué sirvió minimizar el costo con la derivada".

---

## Roadmap completo

Con esta tarea se completan las 6 etapas del roadmap definido en el README. El notebook `descenso_gradiente.ipynb` queda funcional de punta a punta: genera el dataset, entrena con ambos métodos, mide sus métricas y exporta los gráficos comparativos (`curva_convergencia.png`, `comparacion_metodos.png`, `frontera_decision.png`) y la tabla (`metricas.csv`) listos para la diapositiva 12 del PPT.
