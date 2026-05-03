# EXPLANATION.md — Tu Guía Personal del Proyecto

> Este archivo es **solo para ti**. Explica cada archivo, cada decisión de diseño,
> qué hace cada pieza de código y **por qué** lo hice así.
> Léelo antes de presentar el proyecto a un cliente o reclutador — te dará seguridad total.

---

## La Idea Central

Antes de entrar en archivos individuales, el concepto clave:

**El pipeline es un sistema de 9 etapas que transforma un PDF bancario en inteligencia financiera.**

```
PDF crudo
  ↓ [1] pdf_parser.py      — extrae texto del PDF con regex
  ↓ [2] loader.py          — guarda lo crudo en MySQL (staging)
  ↓ [3] categorizer.py     — clasifica cada gasto (Groceries, Dining, etc.)
  ↓ [4] feature_engineering.py — crea columnas derivadas (rolling, lags, fechas)
  ↓ [5] builder.py         — escribe la versión limpia en el data mart
  ↓ [6] stored procedures  — recalcula los agregados mensuales/diarios
  ↓ [7] analysis.py        — genera 9 gráficas automáticamente
  ↓ [8] model_trainer.py   — entrena XGBoost + Prophet + SHAP
  ↓ [9] predictor.py       — predice cuánto gastarás el próximo mes
       ↑ todo orquestado por pipeline.py y scheduler.py
```

---

## Archivos Uno por Uno

---

### `pipeline.py` — El punto de entrada

**¿Qué hace?**
Es el único archivo que llamas directamente con `python pipeline.py`. Es el director de orquesta: llama a todos los módulos en el orden correcto.

**¿Por qué Click?**
Click es la librería estándar para CLIs profesionales en Python. Permite opciones como `--pdf`, `--train-ml`, `--schedule` con validación automática y mensajes de ayuda con `--help`. Alternativa más básica sería `argparse`, pero Click es más limpio.

**¿Por qué `run_pipeline()` es una función separada del `main()`?**
Para que el `scheduler.py` pueda llamar a `run_pipeline(pdf_path)` directamente sin pasar por el CLI. Principio de diseño: separar la lógica del negocio de la interfaz de usuario.

**Los 9 stages con `try/except` individuales:**
Si la EDA falla (por ejemplo, no hay datos suficientes para graficar), el pipeline no se rompe. Los stages opcionales (EDA, ML) están en bloques separados para que los stages críticos (parsing, staging, datamart) siempre corran.

---

### `config/settings.py` — Configuración centralizada

**¿Qué hace?**
Define todas las constantes y configuraciones del proyecto usando Pydantic para validación. Es el único lugar donde existen valores como la URL de la base de datos, rutas de carpetas, o parámetros del modelo XGBoost.

**¿Por qué Pydantic?**
Pydantic valida tipos automáticamente. Si alguien pone `DB_PORT=abc` en el `.env`, Pydantic lanza un error claro en vez de un crash misterioso más tarde. Es el estándar en proyectos de producción (FastAPI lo usa por esto).

**¿Por qué `.url_safe` vs `.url`?**
Para que los logs nunca impriman tu contraseña de MySQL. `url_safe` usa `****` en lugar del password real. Cuando estás debuggeando y miras los logs, no quieres ver credenciales.

**El singleton `settings = Settings()`:**
Se crea una sola instancia al momento de importar el módulo. Todos los demás módulos hacen `from config.settings import settings` y comparten la misma instancia. Esto evita re-leer el `.env` en cada importación.

---

### `config/merchants.py` — El cerebro del categorizador

**¿Qué hace?**
Un diccionario de 15 categorías, cada una con una lista de patrones regex que se comparan contra las descripciones de transacciones. También define colores y etiquetas para las gráficas.

**¿Por qué regex y no ML para categorizar?**
Con ~200-500 transacciones/mes de un solo banco polaco, entrenarte un clasificador ML requeriría cientos de transacciones ya etiquetadas. Los regex son más rápidos, 100% interpretables, y fáciles de extender. Si un día tienes 10,000 transacciones de múltiples bancos, puedes reemplazar este módulo con un fine-tuned text classifier.

**`CATEGORY_PRIORITY` — el orden importa:**
INCOME se verifica primero porque `"ZWROT"` (devolución) aparece tanto en ingresos de impuestos como en descripciones de farmacias. Si verificaras HEALTHCARE primero, una devolución del fisco podría clasificarse como gasto médico. RENT va segundo porque es una palabra corta que podría hacer match dentro de otras descripciones si no se controla el orden.

**¿Cómo extender el categorizador?**
Solo agrega una línea al diccionario en `config/merchants.py`. Por ejemplo, si quieres reconocer "Carrefour":
```python
"GROCERIES": [
    ...
    r"carrefour",  # ← agregar aquí
]
```
Cero cambios en código, cero reinicio de nada.

---

### `src/ingestion/pdf_parser.py` — El corazón del parsing

**¿Qué hace?**
Abre el PDF de Erste Bank, extrae todo el texto con `pdfplumber`, agrupa las líneas en bloques por transacción, y parsea cada bloque en un objeto `RawTransaction`.

**¿Por qué pdfplumber y no PyPDF2?**
PyPDF2 lee el PDF en stream de texto simple y pierde la alineación de columnas. Los PDFs de Erste Bank tienen dos columnas (descripción izquierda, monto derecha), y pdfplumber preserva esa estructura con `x_tolerance=3, y_tolerance=3`. Probado con los 8 páginas del PDF que subiste.

**`_compute_sha256()`:**
Calcula el hash SHA-256 del archivo antes de parsearlo. Este hash se guarda en `stg_pdf_loads` y se verifica antes de cada carga. Si corres el pipeline dos veces con el mismo PDF, el segundo run se salta silenciosamente en vez de duplicar datos.

**`parse_polish_amount("-101,42 PLN") → -101.42`:**
Polonia usa coma como separador decimal y espacio como separador de miles (e.g., `3 157,06`). Esta función:
1. Elimina letras de moneda
2. Elimina espacios (separador de miles)
3. Reemplaza `,` por `.`
4. Convierte a float

**`_group_into_blocks()`:**
Este es el método más delicado. Las transacciones en el PDF de Erste Bank siempre empiezan con una fecha (`26 Apr 2026`) seguida de "Booking date". El parser usa este patrón como delimitador para agrupar líneas en bloques. Cada bloque = una transacción.

**`dataclass RawTransaction`:**
Uso un dataclass (no dict) para la representación intermedia porque los dataclasses dan autocompletado en IDEs, documentación implícita de qué campos existen, y conversión directa a dict con `.__dict__`.

---

### `src/staging/loader.py` — La capa de staging

**¿Qué hace?**
Toma el DataFrame parseado y lo escribe en las tablas crudas de MySQL (`stg_raw_transactions`). Maneja la idempotencia, crea el audit log, e inserta en batches para eficiencia.

**¿Por qué existe una capa de staging separada del data mart?**
Principio de data engineering: separar los datos crudos de los transformados. Si mañana cambias la lógica del categorizador (por ejemplo, reclasificas "ROSSMANN" de BEAUTY a HEALTHCARE), puedes re-procesar desde staging sin re-parsear el PDF. El staging es tu safety net.

**`engine.begin()` vs `engine.connect()`:**
`begin()` abre una transacción automática. Si falla cualquier INSERT en el batch, hace ROLLBACK automático y no quedan datos parciales en la tabla. `connect()` sin `begin()` no tiene esa garantía. Siempre usa `begin()` para escrituras.

**Batch de 500 rows:**
Insertar fila por fila es extremadamente lento en MySQL por el overhead de cada roundtrip de red. Insertar en lotes de 500 reduce ese overhead en ~100x. El número 500 es empírico: suficientemente grande para ser eficiente, suficientemente pequeño para no exceder el `max_allowed_packet` de MySQL.

---

### `src/transform/categorizer.py` — El categorizador

**¿Qué hace?**
Toma una descripción de transacción y retorna un código de categoría como `"GROCERIES"` o `"SUBSCRIPTIONS"`.

**`re.compile()` en `__init__`, no en `categorize()`:**
Si compilaras los regex dentro de `categorize()`, los compilarías en cada llamada. Para un PDF de 200 transacciones con 80 patrones, eso son 16,000 compilaciones innecesarias. Compilar una vez en `__init__` y reusar es la práctica correcta.

**`@lru_cache(maxsize=2048)`:**
LRU (Least Recently Used) cache de Python. Si la misma descripción aparece dos veces (mismo Biedronka, mismo ROSSMANN), el segundo call retorna el resultado del primero sin ejecutar los regex. En un PDF típico, muchas transacciones tienen la misma tienda. El cache puede reducir el trabajo real en 50-70%.

**`explain(description)`:**
Función de debug que te dice exactamente qué patrón matcheó (o cuáles verificó sin match). Útil cuando quieres entender por qué una transacción cayó en "OTHER" y necesitas agregar un patrón nuevo.

---

### `src/transform/feature_engineering.py` — El motor de features

**¿Qué hace?**
Toma el DataFrame categorizado y agrega ~15 columnas derivadas: features de fecha, rolling windows, features de balance, features de lag para ML.

**¿Por qué rolling windows sobre daily-resampled y no sobre filas?**
Imagina que el 1 de abril tienes 5 transacciones y el 2 de abril tienes 0. Si calculas el rolling 7d sobre filas (no fechas), el "7 días" en realidad podría ser 3 días de calendario. Resampleas a serie diaria primero, calculas el rolling, y luego mapeas de vuelta por fecha. Así `rolling_7d_spend` siempre son literalmente los últimos 7 días calendario.

**`create_monthly_ml_features()`:**
Produce una tabla de una fila por (año, mes, categoría) con columnas `lag_1m_spend`, `lag_2m_spend`, `lag_3m_spend`. Esto es lo que XGBoost necesita para aprender patrones: "el mes pasado gasté X en groceries, ¿cuánto gasto este mes?".

**`days_since_payday`:**
Heurística basada en que la mayoría de salarios polacos llegan entre el 1 y el 5 del mes. El día 3 se usa como referencia. Esta feature ayuda al modelo a entender que los gastos tienden a ser más altos los primeros días del mes (efecto paycheck).

---

### `sql/01_staging_schema.sql` — Staging DDL

**¿Qué hace?**
Define las dos tablas de staging: `stg_pdf_loads` (audit log) y `stg_raw_transactions` (datos crudos).

**`UNIQUE KEY uq_file_hash`:**
La restricción de unicidad sobre `file_hash` en `stg_pdf_loads` es la segunda línea de defensa contra duplicados (la primera es el check en Python). Incluso si el código Python fallara el check, MySQL rechazaría insertar el mismo hash dos veces.

**`ENGINE=InnoDB`:**
InnoDB en lugar de MyISAM porque soporta transacciones (ACID), foreign keys, y row-level locking. MyISAM bloquea la tabla completa en cada escritura — catastrófico en producción.

**`DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci`:**
Los nombres de tiendas polacas tienen caracteres `ż`, `ź`, `ć`, `ą`, `ś`, `ę`, `ó`, `ł`, `ń`. `utf8mb4` es la única codificación que los maneja correctamente en MySQL (el `utf8` estándar de MySQL es en realidad UTF-8 de 3 bytes y puede romper caracteres poliacos y emojis).

---

### `sql/02_datamart_schema.sql` — Data Mart DDL

**Fact table + dimension table = Star Schema:**
`fact_transactions` tiene todas las métricas (amounts, balances) y claves. `dim_categories` tiene los descriptores de categorías. Esto es el patrón estándar de data warehousing. Power BI lo importa nativamente y crea las relaciones automáticamente.

**Columnas de fecha desnormalizadas:**
`year`, `month`, `day`, `day_of_week`, `quarter`, `is_weekend` están directamente en `fact_transactions` en vez de en una `dim_date` separada. Para este tamaño de dataset (~10K transacciones/año), la desnormalización simplifica las queries y elimina JOINs innecesarios en Power BI.

**Las tres vistas (`CREATE OR REPLACE VIEW`):**
`v_monthly_cashflow`, `v_category_totals`, `v_merchant_leaderboard` son consultas preescitas que Power BI puede importar directamente. Cuando cambias datos, las vistas siempre retornan los valores actualizados sin que tengas que hacer nada.

---

### `sql/03_stored_procedures.sql` — Stored Procedures

**`sp_refresh_monthly_aggregates()`:**
Recalcula `agg_monthly_spending` para los meses que tienen datos nuevos. Usa `ON DUPLICATE KEY UPDATE` para ser idempotente: correrlo 10 veces produce el mismo resultado que correrlo 1 vez.

**`DECLARE EXIT HANDLER FOR SQLEXCEPTION BEGIN ROLLBACK`:**
Si cualquier query dentro del procedure falla, hace ROLLBACK automático y re-lanza el error. Esto evita estados inconsistentes donde los agregados están parcialmente actualizados.

**¿Por qué usar stored procedures en vez de Python?**
Los aggregates sobre millones de rows son más rápidos corriendo dentro del motor de MySQL que leyendo todos los datos a Python, agregando en pandas, y escribiendo de vuelta. Para este proyecto es una diferencia menor, pero es la práctica correcta y escala bien.

---

### `src/datamart/builder.py` — Constructor del Data Mart

**`INSERT IGNORE INTO fact_transactions`:**
`INSERT IGNORE` con el `UNIQUE KEY (raw_id)` hace que re-correr el pipeline no duplique datos. Si un `raw_id` ya existe en la tabla, la fila se ignora silenciosamente. Alternativa sería `INSERT ... ON DUPLICATE KEY UPDATE`, pero como queremos inmutabilidad en el data mart, IGNORE es más apropiado.

**`_load_category_map()`:**
Carga el mapeo `category_code → category_id` al crear la instancia. Esto evita hacer un SELECT por cada transacción cuando se construye el batch. Con 200 transacciones, sería 200 queries innecesarios.

---

### `src/eda/analysis.py` — Motor de EDA

**`matplotlib.use("Agg")`:**
Sin esta línea, matplotlib intenta abrir una ventana gráfica. En un servidor Linux sin display (o cuando corres `--schedule`), esto lanza un error. El backend "Agg" renderiza a archivo sin necesitar display.

**Dark theme (`#0F1117` background):**
Escogí colores que coinciden con Power BI Dark Mode y los dashboards modernos. Hace que las capturas de pantalla del proyecto se vean profesionales para el portfolio.

**`fig.get_facecolor()` en `savefig`:**
Si no pasas `facecolor` explícitamente a `savefig`, matplotlib puede guardar con fondo blanco aunque hayas configurado `figure.facecolor`. Esta línea asegura que el color de fondo sea consistente.

**`console.print()` con Rich:**
La tabla de resumen al final del EDA usa Rich para formateo de terminal con colores y alineación. Cuando muestras el proyecto en una demo, la salida del terminal se ve profesional.

---

### `src/ml/model_trainer.py` — Entrenamiento ML

**¿Por qué XGBoost + Prophet en vez de solo uno?**

| Aspecto | XGBoost | Prophet |
|---------|---------|---------|
| Fortaleza | Features tabulares, interacciones | Tendencia temporal, estacionalidad |
| Necesita | Lag features, categoría codificada | Solo columna de fecha y valor |
| Output | Punto estimado | Punto + intervalo de confianza |
| Maneja meses faltantes | No directamente | Sí, automáticamente |

Juntos se complementan: XGBoost da la mejor estimación puntual, Prophet da los bounds de confianza.

**`TimeSeriesSplit` para cross-validation:**
`KFold` estándar mezcla datos aleatoriamente, creando data leakage en series temporales. Si en el fold de validación tienes datos de marzo y en training tienes datos de abril, el modelo "aprende del futuro". `TimeSeriesSplit` garantiza que siempre entrenas en el pasado y validas en el futuro.

**SHAP (SHapley Additive exPlanations):**
SHAP responde: "¿Por qué el modelo predijo 800 PLN en groceries para mayo?" No es un black box. Para un portfolio, poder explicar las predicciones del modelo ML es la diferencia entre parecer un analista junior y uno senior.

**`warnings.filterwarnings("ignore")`:**
Prophet usa Stan internamente y lanza muchos warnings de compilación en algunos sistemas. Estos no indican errores reales, pero ensucian el output. Los suprimimos globalmente en este módulo.

---

### `src/ml/predictor.py` — Inferencia

**Separación trainer/predictor:**
El trainer es para training (corre raramente, Sundays 03:00). El predictor es para inference (corre en cada pipeline run). Separarlos significa que puedes hacer predicciones sin re-entrenar, y que el código de inference es simple de mantener.

**`sorted(glob("xgb_spending_*.pkl"))[-1]`:**
Los archivos tienen timestamp en el nombre (`xgb_spending_20260430_1430.pkl`). `sorted()` ordena alfabéticamente, y como el formato es `YYYYMMDD_HHMM`, el orden alfabético coincide con el cronológico. El `[-1]` toma el más reciente.

**Iterative forecasting para XGBoost:**
Para predecir 3 meses hacia adelante, XGBoost necesita lags del pasado. Para el mes 2, el "mes 1 real" no existe (es el futuro), así que usamos la predicción del mes 1 como lag. Esto se llama *iterative forecasting* y es la estrategia estándar cuando el modelo no tiene arquitectura recurrente.

**`max(0, predicted)`:**
XGBoost puede predecir valores negativos (matemáticamente posible en regresión lineal). Para gastos, un valor negativo no tiene sentido. Truncamos en 0.

---

### `src/automation/scheduler.py` — Automatización

**APScheduler + Watchdog juntos:**
- **Watchdog** es event-driven: reacciona instantáneamente cuando un PDF nuevo aparece en la carpeta.
- **APScheduler** es time-driven: verifica la carpeta cada N minutos como fallback.

¿Por qué ambos? Si el scheduler estaba apagado cuando pusiste el PDF, el scan del intervalo lo procesará en el próximo ciclo. Si el scheduler está corriendo, watchdog lo procesa inmediatamente.

**`_processing = set()`:**
Algunos filesystems (especialmente Linux con inotify) lanzan múltiples eventos para un mismo archivo (IN_CREATE, IN_MOVED_TO, IN_CLOSE_WRITE). Sin el set, el pipeline correría 2-3 veces en el mismo PDF. El set actúa como mutex.

**`time.sleep(2)` antes de procesar:**
Cuando un archivo se está copiando, el evento `on_created` se dispara cuando el archivo se crea (no cuando termina de escribirse). Sin el delay, el parser podría intentar leer un archivo parcialmente escrito.

**Email en HTML:**
El template de email HTML usa colores del dark theme del proyecto para consistencia visual. Si en una demo muestras el email de notificación, coincide con el resto del proyecto.

---

### `tests/test_pipeline.py` — Suite de Tests

**¿Por qué pytest y no unittest?**
pytest tiene mejor output, fixtures más limpias, y descubrimiento automático de tests. `unittest` requiere más boilerplate. El 95% de los proyectos Python modernos usan pytest.

**Estructura de clases de test:**
Cada clase cubre un módulo. `setup_method()` se corre antes de cada test para evitar estado compartido entre tests.

**`StagingLoader.__new__(StagingLoader)` para tests sin DB:**
El `__init__` de `StagingLoader` intenta conectarse a MySQL. Para testear métodos helper sin una DB real, usamos `__new__()` que crea la instancia sin llamar `__init__`. Es un patrón de testing avanzado que evita necesitar un MySQL real en el CI/CD.

**Tests de integración al final:**
`TestIntegrationChain` prueba el chain completo parse→categorize→feature_engineer con datos sintéticos. No necesita DB ni PDF real, pero verifica que los módulos se conecten correctamente entre sí.

---

## Cómo Explicar el Proyecto en una Entrevista

**Pregunta: "¿Qué hace tu proyecto?"**
> "Es un pipeline de data engineering end-to-end que procesa extractos bancarios en PDF, los ingesta en un data warehouse MySQL de dos capas, aplica feature engineering para análisis temporal, genera visualizaciones automáticas, y usa XGBoost con Prophet para predecir el gasto mensual por categoría. Está completamente automatizado con un scheduler que detecta nuevos PDFs y los procesa sin intervención."

**Pregunta: "¿Por qué dos capas en MySQL?"**
> "Separé staging de data mart para auditoría y re-procesabilidad. Si cambio la lógica de categorización, puedo re-transformar desde el staging sin volver a parsear el PDF. Es el principio de Data Vault: los datos crudos son inmutables."

**Pregunta: "¿Por qué dos modelos de ML?"**
> "XGBoost es mejor con features tabulares y lag variables — aprende interacciones entre categoría, mes del año, y gastos previos. Prophet es mejor para la componente temporal pura: trend y estacionalidad sin features adicionales. Los combino: XGBoost da la estimación puntual, Prophet da los intervalos de confianza. SHAP hace todo explicable."

**Pregunta: "¿Cómo garantizas idempotencia?"**
> "Tres capas: SHA-256 del archivo en Python antes de cargar, UNIQUE constraint en la columna `file_hash` en MySQL, y `INSERT IGNORE` con `UNIQUE KEY (raw_id)` en el data mart. Un mismo PDF puede correr 100 veces sin duplicar datos."

---

## Tips para el Portfolio en GitHub

1. **Pin este repo** en tu perfil de GitHub
2. **El README.md** tiene badges, arquitectura ASCII, y tabla de tech stack — eso llama la atención en los primeros 5 segundos
3. **Capturas de pantalla** de las gráficas generadas en `reports/figures/` — agrégalas al README con `![Chart](reports/figures/01_monthly_cashflow.png)`
4. **No subas tu `.env`** ni tus PDFs bancarios — el `.gitignore` ya los excluye
5. Menciona en el README que los datos son **reales de tu propio banco** — eso da credibilidad al proyecto

---

*EXPLANATION.md — Finance Intelligence Pipeline v1.0 — Para uso personal del autor*
