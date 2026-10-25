# RabbitSense

Prototipo de investigación para seguir a **Cinna** en vídeo y estimar **candidatos** de movimiento. **No** diagnostica estrés, dolor ni bienestar. Las cajas de V3.2 son anotaciones manuales/interpoladas, **no** detecciones automáticas.

## Estado

- **v36**: referencia actual; 20 candidatos de locomoción, 43 de poco movimiento y 21 desconocidos en IMG_5177 (84 muestras). **No hay etiquetas de verdad-terreno que validen precisión.**
- **v37**: experimento de consistencia con flujo óptico del cuerpo; 1 candidato de locomoción, 43 de poco movimiento y 40 desconocidos. **No es una mejora validada.**
- **v32**: generación de trayectorias a partir de marcas; conservada en `scripts/legacy/`.

## Instalación (Mac Apple Silicon)

```bash
cd RabbitSense_repo
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

## Ejecutar

Conserva tus datos existentes y coloca el vídeo en `data/cinna_review_v24/IMG_5177.mp4` y el CSV en `runs/v32/cinna_trayectoria_v32.csv` (o usa rutas absolutas a los originales).

```bash
.venv/bin/python -m rabbitsense --version v36 \
  --video data/cinna_review_v24/IMG_5177.mp4 \
  --tracks runs/v32/cinna_trayectoria_v32.csv \
  --out runs/predictions_v36/5177 --video-out
```

Para probar v37 cambia `--version v37` y `--out runs/predictions_v37/5177`.

## Estructura

- `src/rabbitsense/`: implementaciones v36 (referencia) y v37 (experimental), ejecutables con `python -m rabbitsense`.
- `scripts/legacy/`: algoritmos v32, v34, v35 para reproducibilidad; no son la API principal.
- `data/`: vídeos y anotaciones locales, ignorados por Git.
- `runs/`: predicciones, modelos y vídeos generados, ignorados por Git.
- `tests/`: pruebas básicas.

## Limitaciones

No se ha medido exactitud contra una validación humana de locomoción. El flujo óptico puede incluir fondo dentro de la caja. Las versiones antiguas no se han refactorizado ni garantizado como compatibles. No subir vídeos privados, modelos pesados ni entornos virtuales al repositorio.
