"""Smoke test del proyecto real (no de la maqueta)."""
import asyncio, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from graph import graph_builder
from modules.memory.cache import init_cache
from modules.memory.long_term import get_vector_store
from settings import settings
from tools.tracing import flush, traza_conversacion

async def main():
    fallos = []
    await init_cache()
    vs = get_vector_store(); vs.clear()
    async with AsyncPostgresSaver.from_conn_string(settings.URI_PSYCOPG) as saver:
        await saver.setup()
        app = graph_builder.compile(checkpointer=saver)
        # Hilo NUEVO en cada pasada: reutilizar el mismo hacia que el historial
        # arrastrara las respuestas de pasadas anteriores y el modelo se copiara
        # a si mismo, dando siempre el mismo fallo.
        cfg = {"configurable": {"thread_id": f"smoke-{int(time.time()*1000)}"}}
        async def t(txt, espera):
            print(f"\n  > {txt}")
            ini = time.time()
            with traza_conversacion(entrada=txt, sesion=cfg["configurable"]["thread_id"]) as traza:
                r = await app.ainvoke({"messages":[HumanMessage(content=txt)]}, cfg)
                if traza is not None:
                    traza.update(output=r["messages"][-1].content)
            print(f"  < {r['messages'][-1].content[:200]}")
            print(f"    rama={r.get('workflow')} ({time.time()-ini:.1f}s)")
            if r.get("workflow") != espera:
                fallos.append(f"'{txt[:30]}' fue a {r.get('workflow')}, se esperaba {espera}")
            return r
        await t("Hola, me llamo Ana y soy de Sevilla.", "conversation")
        r = await t("Te acuerdas de donde vivo?", "conversation")
        if "sevilla" not in r["messages"][-1].content.lower():
            fallos.append("no recordo la ciudad")
        r = await t("Dibujame un gato con gafas de sol", "image")
        if not r.get("image_path"): fallos.append("no genero imagen")
        prev_ai = r["messages"][-1].content  # lo ultimo que dijo antes del audio
        r = await t("Mandame un audio con tu voz", "audio")
        if not r.get("audio_buffer"): fallos.append("no genero audio")
        else:
            txt = r["messages"][-1].content
            bajo = txt.lower()
            if "/" in txt or ".mp3" in bajo or ".wav" in bajo:
                fallos.append(f"el audio devolvio una ruta: {txt[:60]}")
            elif any(x in bajo for x in ("no puedo enviar", "no puedo mandar",
                                         "no puedo enviarte", "no puedo grabar")):
                fallos.append(f"NIEGA poder mandar audio mientras lo manda: {txt[:70]}")
            elif len(txt.split()) < 3:
                fallos.append(f"respuesta de voz degenerada: {txt}")
            elif txt[:60].lower() == prev_ai[:60].lower():
                fallos.append(f"la locucion REPITE la respuesta anterior: {txt[:60]}")
        print(f"\n  recuerdos en pgvector: {vs.count()}")
    # Lo que quede en el buffer de LangFuse se envia antes de salir.
    flush()
    print("\n" + ("FALLOS:\n  " + "\n  ".join(fallos) if fallos else "SMOKE OK"))
    return 1 if fallos else 0

raise SystemExit(asyncio.run(main()))
