# Cómo se construyó este proyecto

El enunciado del módulo 20 pide documentar el **proceso de generación** de cada fichero, no
solo el resultado. Esto es ese registro: qué se hizo, en qué orden, qué falló y qué se aprendió.

Escrito sobre la marcha. Lo que aquí se cuenta como medido, se midió de verdad.

---

## 0 · Antes de escribir código

Se leyó el temario completo (117 PDF) y se analizó el enunciado. Dos hallazgos que
condicionaron todo:

1. **El módulo 20 no es un brief abierto**: es un tutorial prescriptivo, fichero a fichero,
   con `git commit` en cada paso. Eso fija la arquitectura; lo que queda libre es el *cómo*.
2. **Pide menos que el máster**: nada de RAG documental, fine-tuning, MLOps ni CI/CD. Es el
   capstone de los módulos 4, 5, 6, 9 y 12.

Después llegó una restricción del cliente (yo mismo): **coste cero, ni un euro**. El enunciado
usa cinco servicios de pago. Buscar sustituto libre para los cinco pasó a ser el eje del
proyecto, y de paso cubre el módulo 7 (modelos open source) y convierte el módulo 1
(selección de modelo) en una decisión real con números.

---

## 1 · Entorno

**Decisiones, con su porqué:**

| Se pedía | Se usó | Motivo |
|---|---|---|
| conda | **uv** | El propio enunciado pide `uv.lock` al final. Un binario, sin permisos de admin |
| Python 3.11 | **Python 3.13** | Se midió antes de decidir (ver abajo) |
| Homebrew + compilar pgvector | **Podman** + imagen oficial | El enunciado asume macOS; en Windows no aplica |

**La medición de Python.** Había recomendado bajar a 3.11 "porque Chainlit va por detrás". Era
una precaución sin dato. Un `uv pip compile` de un minuto lo resolvió: el stack entero resuelve
e instala en 3.13.15, y los 18 módulos importan sin un fallo. La cautela era mía, no del
ecosistema.

**Precio que se paga:** `torch-directml` solo publica ruedas hasta cp312, así que la generación
de imagen va en CPU. Se acepta: 9-18 s por imagen es tolerable en un chat.

---

## 2 · El stack libre

Cada sustitución se verificó **antes** de construir sobre ella:

| Pieza | Prueba que se hizo | Resultado |
|---|---|---|
| `qwen3:14b` | Router con salida estructurada, 4 mensajes distintos | **4/4**, 0,3 s por decisión |
| `gemma3:4b` | Leer una captura sintética con texto y números | Leyó todo, incluidos los valores |
| `faster-whisper` | Transcribir un WAV generado por Piper | Texto idéntico al original |
| Piper | Sintetizar una frase en español | 2,5 s de audio en 0,14 s |
| `sd-turbo` | Generar 512×512 y **describirla con el modelo de visión** | *"un gato astronauta flotando en el espacio"* |

La última es la prueba que más dice: **generación y comprensión locales validándose entre sí**.

---

## 3 · Maqueta antes que producto

Antes de escribir los 20 ficheros del enunciado se montó una **maqueta** con la topología
exacta del grafo (`notebooks/prueba_funcional.py`), para comprobar que las piezas encajaban.
Costó 55 s ejecutarla y ahorró rehacer medio proyecto.

De ahí salieron tres cosas que se aplicaron al código definitivo:

### El umbral de deduplicación se mide, no se elige

Puesto a ojo en 0,15, no deduplicaba nada. Medido con `nomic-embed-text`:

| Par | Distancia |
|---|---|
| *"Se llama Ana"* / *"Su nombre es Ana"* | 0,287 |
| *"Se llama Ana"* / *"El usuario se llama Ana"* | 0,157 |
| *"Vive en Sevilla"* / *"Reside en Sevilla"* | 0,140 |
| *"Se llama Ana"* / *"Vive en Sevilla"* | 0,509 |
| *"Se llama Ana"* / *"Le interesa la ciberseguridad"* | 0,498 |

Hueco entre 0,29 y 0,50. **0,30** cae dentro, pero el margen es estrecho: la reformulación
más alejada de un mismo hecho (*"Se llama Ana"* / *"Su nombre es Ana"*) se queda en 0,287, a
trece milésimas del umbral. Con frases más largas convendría volver a medir antes de subirlo.

### En Windows, `psycopg` async no arranca

```
psycopg.InterfaceError: Psycopg cannot use the 'ProactorEventLoop' to run in async mode
```

Apareció en la maqueta, y se anotó que volvería a aparecer en la API. Volvió. Ver §6.

### Los prompts del enunciado no son paranoia

Se simplificaron por parecer redundantes y **reprodujeron exactamente los fallos que evitan**.
Detallado en §5.

---

## 4 · Los ficheros

Orden de construcción y qué tiene de particular cada uno:

| Fichero | Qué tiene de particular |
|---|---|
| `settings.py` | `URI_PSYCOPG` como propiedad: SQLAlchemy quiere `+asyncpg`, el checkpointer lo quiere sin él |
| `core/prompts.py` | Todos los prompts juntos y versionados. Cada regla lleva su motivo |
| `core/exceptions.py` | Una excepción por módulo: saber qué se rompió sin leer la traza |
| `tools/tools.py` | Logger que no duplica handlers al reimportarse |
| `tools/tracing.py` | `@observe` **degradable**: sin claves de LangFuse es un decorador que no hace nada. Así el proyecto arranca sin cuenta en ningún sitio |
| `modules/speech/` | Modelos cargados con `lru_cache`: pesan segundos la primera vez |
| `modules/image/text_to_image.py` | Escena → prompt mejorado → imagen. Si el LLM devuelve un JSON malo, se degrada al historial en vez de romper |
| `modules/image/image_to_text.py` | Imágenes al modelo de visión; PDF y texto se extraen y se resumen. No se manda un PDF entero a un modelo de visión |
| `modules/memory/long_term/vector_store.py` | Singleton, índice HNSW, y deduplicación antes de insertar |
| `modules/memory/cache.py` | Solo cachea la rama de conversación: una imagen o un audio producen un fichero nuevo cada vez |
| `graph/state.py` | `image_path` y `audio_buffer` en el **estado**, no en el mensaje. Ver §5 |
| `graph/utils/chains.py` | Tres cadenas: router, personaje y **locución con salida estructurada** |
| `graph/nodes.py` | Siete nodos, todos instrumentados con `@observe` |
| `app.py` | Chainlit. El audio del micrófono llega como PCM crudo y hay que envolverlo en WAV |
| `main.py` | FastAPI con su propio arranque, por lo del bucle de eventos |

---

## 5 · Los fallos que enseñaron algo

### El bot que niega poder hacer lo que está haciendo

Al pedirle un audio, respondía **"Lamento decirte que no puedo enviar audios"**… mientras Piper
le sintetizaba la voz. La regla estaba en la ficha de personaje y aun así la ignoraba: su
prior de *"soy un modelo de texto"* gana.

Se probaron cuatro enfoques, cinco peticiones cada uno:

| Enfoque | Aciertos | Qué fallaba |
|---|---|---|
| Afirmar la capacidad | 3/5 | Colaba acotaciones `[Audio: "..."]` |
| Reformular como guion | 2/5 | Se contradecía a media frase |
| Guion con ejemplos | 1/5 | **Copiaba los ejemplos literalmente** |
| **Salida estructurada** | **5/5** | — |

Un esquema Pydantic no deja sitio para el meta-comentario. Es la misma técnica que hace que el
router acierte 4/4. **Cuando el formato importa, se impone con un esquema, no con una súplica.**

### La ruta que contamina el historial

Corregido lo anterior, el modelo empezó a inventarse rutas: `/audio_1789681423.mp3`, que la voz
leía como *"barra audio uno punto siete ocho nueve…"*. La causa no era el prompt: el nodo de
imagen metía el nombre del fichero **en el texto del mensaje**, el modelo lo veía en el
historial y copiaba el patrón.

**La ruta del artefacto va en el estado. Siempre.** Por eso el enunciado tiene `image_path` y
`audio_buffer` como campos de `AgentState`: el fichero lo adjunta la capa de presentación, no
el LLM.

### Defensa en profundidad para la voz

Aun con todo lo anterior, a veces sale *"aquí tienes un audio diciendo: '…'"*. Confiar solo en
el prompt es frágil, así que el módulo de voz **limpia el texto antes de pronunciarlo**:
quita anuncios, comillas y marcas. Probado con 5 casos, 5/5.

### Un test verde que no valía nada

La primera versión del smoke daba **SMOKE OK** con un audio que decía *"no puedo mandar
audios"*, porque solo comprobaba que el WAV existiera. **Un test que no mira el contenido no
es un test.** Ahora comprueba que no niegue, que no devuelva una ruta y que no sea degenerado.

### El modelo se copia a sí mismo

Durante días el mismo fallo salía **con el texto exacto**, aunque la temperatura fuera 0,5.
No era varianza: el smoke reutilizaba siempre `thread_id="smoke"` y el checkpointer había
acumulado **97 estados**. La respuesta mala estaba en el historial y el modelo la repetía.
Ahora cada pasada usa un hilo nuevo.

### Probando contra un servidor fantasma

Tres rondas de depuración de la API dieron el mismo error después de arreglarlo. El proceso
viejo seguía dueño del puerto y `pkill -f "main.py"` no casaba con `-m uvicorn main:app`.
**Antes de depurar, comprobar contra qué se está probando.**

---

## 6 · La integración

| Paso | Qué se hizo |
|---|---|
| PostgreSQL + pgvector | Contenedor Podman con la imagen oficial, publicado solo en `127.0.0.1` |
| Memoria corta → Postgres | `AsyncPostgresSaver`. Aquí volvió lo del bucle de eventos |
| Memoria larga → Postgres | `PostgreSQLVectorStore` con índice HNSW y deduplicación |
| Caché de respuestas | Tabla con SQLAlchemy y caducidad, solo para la rama de conversación |
| LangFuse | `@observe` en los siete nodos, **degradable** si no hay claves |
| FastAPI | Cinco endpoints, con la imagen y el audio en base64 |

**El bucle de eventos, resuelto del todo.** Tres intentos:

1. Fijar la política al importar el módulo → **no vale**: con `uvicorn main:app` el bucle ya
   existe cuando se importa.
2. `uvicorn.run(..., loop="asyncio")` tras fijar la política → **no vale**: uvicorn reimpone
   el suyo.
3. `uvicorn.Config(loop="none")` + `asyncio.run(server.serve(), loop_factory=SelectorEventLoop)`
   → **funciona**. El bucle lo creamos nosotros.

---

## 7 · Qué quedó comprobado

- El grafo recorre sus tres ramas y el resumen se dispara solo.
- La memoria larga guarda, deduplica y recupera desde pgvector.
- Los cinco endpoints responden, incluidos los de subida de fichero.
- La visión lee capturas reales; la transcripción entiende audio real.
- Coste de operación: **0 €**.

---

## 8 · Lo que no se hizo, y por qué

- **No se usó ningún servicio de pago**, aunque el enunciado los nombre. La arquitectura es
  idéntica; solo cambia el proveedor dentro de `modules/`, que es justo la capa que existe
  para eso.
- **No se activó LangFuse por defecto.** Requiere una cuenta y el proyecto tenía que arrancar
  sin ninguna. El código está instrumentado y basta rellenar dos variables.
- **No se subió a producción.** El enunciado no lo pide y no había dónde.
