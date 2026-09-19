# Agente Multimodal

Asistente conversacional construido con **LangGraph** que habla, escucha, ve, dibuja y
recuerda. Proyecto final del máster **AI Engineer**.

**Todo corre en local y no cuesta nada.** Ni una clave de pago, ni un dato saliendo del
ordenador.

---

## Qué hace

| | |
|---|---|
| 💬 **Conversa** | Mantiene el hilo y lo resume solo cuando se alarga |
| 🎙️ **Escucha** | Transcribe tu voz con Whisper |
| 🔊 **Habla** | Te responde con voz sintetizada |
| 👁️ **Ve** | Entiende imágenes y extrae el contenido de PDF y documentos |
| 🎨 **Dibuja** | Genera imágenes a partir de la conversación |
| 🧠 **Recuerda** | Aprende cosas de ti y las recupera en conversaciones futuras |

---

## Arranque rápido

### 1. Requisitos

- **Python 3.13** y [uv](https://docs.astral.sh/uv/)
- **Podman** o Docker
- **[Ollama](https://ollama.com)** corriendo en local

```bash
ollama pull qwen3:14b          # texto, router, memoria
ollama pull gemma3:4b          # visión
ollama pull nomic-embed-text   # embeddings
```

### 2. Base de datos

```bash
podman run -d --name agente-pg \
  -e POSTGRES_USER=root -e POSTGRES_PASSWORD=password -e POSTGRES_DB=agent \
  -p 127.0.0.1:5432:5432 -v agente-pgdata:/var/lib/postgresql/data \
  docker.io/pgvector/pgvector:pg17
```

Para volver a levantarla tras reiniciar: `podman machine start && podman start agente-pg`.

### 3. Entorno

```bash
uv venv --python 3.13
uv pip install -r requirements.txt
cp .env.template .env          # opcional: los valores por defecto ya funcionan
```

### 4. Arrancar

```bash
chainlit run app.py -w         # interfaz de chat  -> http://localhost:8000
python main.py                 # API REST          -> http://localhost:8000/docs
```

> [!] **En Windows, la API se arranca con `python main.py`, no con `uvicorn main:app`.**
> El motivo está explicado en *Decisiones técnicas*.

---

## Stack libre

El enunciado del máster usa cinco servicios de pago. Aquí están sustituidos por
equivalentes locales, todos verificados con medidas reales:

| Función | Sustituye a | Se usa | Medido |
|---|---|---|---|
| Texto y razonamiento | OpenAI `gpt-4.1` | Ollama `qwen3:14b` | router 4/4 · 0,3 s/decisión |
| Visión y OCR | OpenAI `gpt-4.1` vision | Ollama `gemma3:4b` | lee capturas completas · ~10 s |
| Voz a texto | OpenAI `whisper-1` | `faster-whisper small` | transcripción exacta · 2,7 s |
| Texto a voz | ElevenLabs | **Piper** `es_ES-davefx` | 2,5 s de audio en 0,14 s |
| Generación de imagen | OpenAI `dall-e-3` | `sd-turbo` (diffusers) | 512×512 en 9-18 s |
| Embeddings | `all-MiniLM-L6-v2` | `nomic-embed-text` | 768 dimensiones |
| Trazabilidad | LangFuse Cloud | LangFuse **opcional** | sin claves, no traza |

**Coste de operación: 0 €.**

---

## Arquitectura

```
START → memory_extraction → router → memory_injection
      → { conversation | image | audio } → (¿resumir?) → END
```

| Nodo | Qué hace |
|---|---|
| `memory_extraction` | Extrae hechos del usuario y los guarda en pgvector |
| `router` | Decide si la respuesta va en texto, imagen o voz (salida estructurada) |
| `memory_injection` | Recupera lo que sabemos y lo mete en el prompt |
| `conversation` | Responde en texto. Es la única rama con caché |
| `image` | Inventa la escena, mejora el prompt y genera la imagen |
| `audio` | Escribe la locución y la sintetiza |
| `summarize` | Resume y recorta el historial al pasar de 20 mensajes |

### Las dos memorias

- **Corta** — el hilo de la conversación, en el *checkpointer* de LangGraph sobre PostgreSQL.
  Sobrevive a cerrar la aplicación.
- **Larga** — hechos sobre el usuario, como vectores en pgvector. Antes de guardar uno nuevo
  se comprueba por similitud que no exista ya: sin eso, repetir algo llena la memoria de ruido.

### Estructura

```
agente-multimodal/
├── app.py                    interfaz de Chainlit
├── main.py                   API de FastAPI
├── settings.py               configuración (pydantic-settings)
├── agents.md                 cómo se construyó cada pieza
├── core/
│   ├── prompts.py            todos los prompts, en un solo sitio
│   └── exceptions.py         una excepción por módulo
├── graph/
│   ├── graph.py              el StateGraph
│   ├── state.py              AgentState
│   ├── nodes.py              los siete nodos
│   ├── edges.py              las aristas condicionales
│   └── utils/
│       ├── chains.py         cadenas LCEL (router, personaje, locución)
│       └── helpers.py        fábricas de modelos y módulos
├── modules/
│   ├── speech/               faster-whisper · Piper
│   ├── image/                gemma3 · sd-turbo
│   └── memory/
│       ├── long_term/        vector store + gestor de memoria
│       └── cache.py          caché de respuestas con caducidad
├── notebooks/                pruebas ejecutables
└── data/                     modelos y datos locales (fuera del repo)
```

---

## API

| Método | Ruta | Para qué |
|---|---|---|
| `GET` | `/` | Estado del servicio |
| `POST` | `/chat/text` | Mensaje de texto |
| `POST` | `/chat/image` | Imagen + pregunta opcional |
| `POST` | `/chat/file` | Documento (PDF, TXT, MD) + pregunta |
| `POST` | `/chat/audio` | Audio: se transcribe y se responde |

```bash
curl -X POST http://localhost:8000/chat/text \
  -H "Content-Type: application/json" \
  -d '{"message":"Hola, me llamo Ana","thread_id":"demo"}'

curl -X POST http://localhost:8000/chat/image \
  -F "file=@captura.png" -F "prompt=Qué ves aquí?" -F "thread_id=demo"

curl -X POST http://localhost:8000/chat/audio \
  -F "file=@pregunta.wav" -F "thread_id=demo"
```

La imagen y el audio vuelven en base64 dentro del JSON.

---

## Pruebas

```bash
python notebooks/smoke_proyecto.py      # recorre las tres ramas del grafo
python notebooks/prueba_funcional.py    # maqueta del stack, pieza a pieza
```

Terminan con código 0 si todo va bien, y con 1 y la lista de fallos si no. Las comprobaciones
son de **contenido**, no solo de que el fichero exista.

### Evaluación

Una prueba de humo dice si el sistema *funciona*. No dice si funciona **bien**. Para eso está
`evaluacion/`, con ground truth y umbrales:

```bash
python evaluacion/evaluar.py
```

| Qué mide | Cómo |
|---|---|
| **Enrutado** | 26 frases con su rama correcta. Se puntúa aparte el subconjunto de **8 trampas**: frases que hablan de dibujar o de audio sin pedir ni una cosa ni otra |
| **Personaje** | 4 ataques (pregunta directa, inyección de prompt, fuga del prompt de sistema, origen del modelo). Doble medida: reglas deterministas y un **juez LLM** |
| **Latencia** | p50 y p95 del router |

Los casos viven en `evaluacion/casos.py`, separados del runner: son el activo que se revisa y
crece. Cada caso sube a LangFuse como una traza con su score, así que dos ejecuciones se
comparan en la interfaz.

Es **suite de regresión**, no solo informe: si una métrica baja de su umbral, termina con
código 1.

```text
enrutado global           96.2 %   umbral  85.0 %  OK
enrutado casos claros    100.0 %
enrutado TRAMPAS          87.5 %   umbral  60.0 %  OK
personaje (reglas)       100.0 %   umbral  75.0 %  OK
personaje (juez LLM)     100.0 %
latencia del router      p50 0.26 s · p95 0.31 s
```

> El único caso que falla es `tra-06`, «¿Cómo suena tu voz?», que el router manda a la rama de
> audio. Se deja **fallando a propósito**: responder esa pregunta hablando es defendible, y
> ajustar el ground truth para que salga un 100 % sería engañarse. Un eval que siempre da
> verde no está midiendo nada.

---

## Por qué los prompts son así

Los prompts tienen reglas que parecen redundantes. No lo son: cada una tapa un fallo concreto
que **se reprodujo al quitarla**.

| Regla | Qué pasa sin ella |
|---|---|
| *"Puedes enviar audios e imágenes"* | Contesta **"no puedo enviar audios"** mientras Piper le sintetiza la voz |
| *"Responde en texto plano, sin meta-comentarios"* | Devuelve `/audio_1789681423.mp3`, que la voz lee como *"barra audio uno siete ocho nueve…"* |
| *"Solo hechos, no peticiones"* | Guarda *"quiere escuchar la voz del usuario"* como si fuera un dato personal |
| *"No inventes nombres de fichero"* | Copia el patrón de ficheros que vea en el historial |

Y una lección de diseño que no es un prompt: **la ruta del artefacto va en el estado
(`image_path`, `audio_buffer`), nunca en el texto del mensaje.** Si se cuela en el historial,
el modelo aprende el patrón y empieza a inventarse rutas para todo lo demás.

---

## Decisiones técnicas

### Python 3.13, no 3.11

El enunciado pide 3.11 por precaución. Se midió: el stack completo resuelve, instala y
**los 18 módulos importan sin un solo fallo** en 3.13.15. La cautela no tenía fundamento.

*Precio*: `torch-directml` no publica ruedas para 3.13, así que la generación de imagen va en
CPU. A 9-18 s por imagen, no molesta. Si algún día molesta, `onnxruntime-directml` sí tiene
rueda para 3.13.

### `uv` en lugar de conda

No es desviarse: el propio enunciado pide `uv.lock`. Usarlo desde el principio cumple ese
requisito de serie, y no necesita permisos de administrador.

### Podman en lugar de compilar pgvector

El enunciado asume macOS (`brew install postgresql` y compilar con `make`). En Windows no
aplica. La imagen oficial `pgvector/pgvector` ya trae la extensión compilada.

### La rama de audio usa salida estructurada

No es un capricho. Con un prompt normal, por explícito que fuera, el modelo negaba poder
mandar audio. Medido sobre cinco peticiones:

| Enfoque | Aciertos |
|---|---|
| Afirmar la capacidad en el prompt | 3/5 (colaba acotaciones `[Audio: ...]`) |
| Reformular como guion | 2/5 |
| Guion con ejemplos | 1/5 (copiaba los ejemplos literalmente) |
| **Salida estructurada** | **5/5** |

Un esquema Pydantic no deja hueco para el meta-comentario, que era justo el problema.

### En Windows, `python main.py` y no `uvicorn main:app`

`psycopg` en modo async **no funciona sobre `ProactorEventLoop`**, que es el de Windows por
defecto. Con `uvicorn main:app` el bucle ya está creado cuando se importa el módulo, así que
fijar la política llega tarde; y `uvicorn.run()` reimpone el suyo. La solución es crear el
bucle nosotros con `loop="none"` y `asyncio.run(..., loop_factory=asyncio.SelectorEventLoop)`.

---

## Limitaciones conocidas

- **El modelo alucina datos.** `qwen3:14b` puede inventarse títulos de libros o referencias. Es
  un modelo de 14B: sirve para conversar, no como fuente de verdad.
- **La generación de imagen tarda** 9-18 s porque va en CPU.
- **Cambiar entre texto y visión cuesta** 8-10 s: con 32K de contexto no caben los dos modelos
  a la vez en 16 GB de VRAM.
- **Sin trazas por defecto.** LangFuse solo se activa si rellenas sus dos claves en el `.env`.

---

## Licencia

MIT.
