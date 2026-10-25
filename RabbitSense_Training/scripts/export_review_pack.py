#!/usr/bin/env python3
"""Exporta vídeos locales de Cinna a ZIPs pequeños para revisión con ChatGPT.

Por vídeo, divide en tramos de duración configurable; cada ZIP contiene:
  - video_reducido.mp4 (movimiento observable, sin audio)
  - fotogramas JPEG con tiempos originales
  - mosaicos para revisión
  - intervalos_4s.csv (plantilla de etiquetado)
  - metadata.json
No altera originales ni modifica modelos/entrenamientos.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageOps
except ImportError as exc:
    raise SystemExit('Falta Pillow. Ejecútalo con .venv/bin/python del proyecto.') from exc

VIDEO_EXTS = {'.mov', '.mp4', '.m4v', '.avi', '.mkv', '.webm', '.mts', '.m2ts', '.3gp'}


def get_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        binary = shutil.which('ffmpeg')
        if not binary:
            raise SystemExit('No encuentro ffmpeg. Usa .venv/bin/python desde RabbitSense_Training.')
        return binary


def duration_seconds(ffmpeg: str, video: Path) -> float:
    ffprobe = shutil.which('ffprobe')
    if ffprobe:
        r = subprocess.run([ffprobe, '-v', 'error', '-show_entries', 'format=duration',
                            '-of', 'default=noprint_wrappers=1:nokey=1', str(video)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            try:
                val = float(r.stdout.strip())
                if math.isfinite(val) and val > 0:
                    return val
            except ValueError:
                pass
    r = subprocess.run([ffmpeg, '-hide_banner', '-i', str(video)], capture_output=True, text=True)
    m = re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)', r.stderr)
    if not m:
        raise RuntimeError(f'No encuentro duración para {video.name}: {r.stderr[-400:]}')
    return int(m[1])*3600 + int(m[2])*60 + float(m[3])


def run(cmd: list[str], label: str) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f'{label} falló: {p.stderr[-1500:]}')


def safe_stem(name: str) -> str:
    ascii_name = name.encode('ascii', 'ignore').decode('ascii')
    base = re.sub(r'[^a-zA-Z0-9_-]+', '_', ascii_name).strip('_')[:42]
    return base or 'video'


def contact_sheets(images: list[Path], outdir: Path, global_times: list[float], per_sheet: int = 16) -> None:
    columns, cell_w, cell_h = 4, 220, 282
    for offset in range(0, len(images), per_sheet):
        segment = images[offset: offset + per_sheet]
        count_rows = (len(segment) + columns - 1)//columns
        canvas = Image.new('RGB', (columns*cell_w, count_rows*cell_h), '#fafafa')
        draw = ImageDraw.Draw(canvas)
        for j, path in enumerate(segment):
            index = offset + j
            x = (j%columns)*cell_w
            y = (j//columns)*cell_h
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                thumb = ImageOps.contain(im, (208, 245))
            canvas.paste(thumb, (x + (cell_w-thumb.width)//2, y + 4))
            t = global_times[index]
            text = f'{int(t//60):02d}:{t%60:04.1f}'
            draw.text((x+9, y+251), text, fill='#111111')
        canvas.save(outdir/f'mosaico_{offset//per_sheet+1:02d}.jpg', quality=84, optimize=True)


def export_video(source: Path, input_root: Path, out_root: Path, ffmpeg: str, args: argparse.Namespace):
    length = duration_seconds(ffmpeg, source)
    digest = hashlib.sha256(str(source.resolve()).encode('utf-8')).hexdigest()[:8]
    name = f'{safe_stem(source.stem)}_{digest}'
    total_parts = math.ceil(length / args.chunk_seconds)
    written = []
    for part in range(total_parts):
        start = part*args.chunk_seconds
        span = min(args.chunk_seconds, length-start)
        if span < .35:
            continue
        zip_name = f'{name}_parte_{part+1:03d}.zip'
        destination = out_root/zip_name
        if destination.is_file() and not args.overwrite:
            print('  Ya existe (se conserva):', destination.name)
            written.append(destination)
            continue
        with tempfile.TemporaryDirectory(prefix='cinna_export_') as tmp:
            tmp_dir = Path(tmp)
            clip_out = tmp_dir/'video_reducido.mp4'
            # El pad evita el problema de alto impar que ya ocurrió con libx264.
            vf = (f'fps={args.fps},scale={args.size}:{args.size}:'
                  'force_original_aspect_ratio=decrease,'
                  'pad=ceil(iw/2)*2:ceil(ih/2)*2,setsar=1')
            cmd = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                   '-ss', f'{start:.3f}', '-i', str(source), '-t', f'{span:.3f}',
                   '-map', '0:v:0', '-vf', vf, '-an', '-c:v', 'libx264',
                   '-preset', 'veryfast', '-crf', str(args.crf), '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', str(clip_out)]
            run(cmd, 'Comprimir vídeo')
            if not clip_out.exists() or clip_out.stat().st_size < 100:
                raise RuntimeError('ffmpeg creó un MP4 vacío: '+source.name)

            frames_dir = tmp_dir/'fotogramas'
            frames_dir.mkdir()
            # Secuencia indexada cada N segundos del vídeo comprimido.
            sample_filter = f'fps=1/{args.sample_seconds}'
            cmd = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                   '-i', str(clip_out), '-vf', sample_filter,
                   '-q:v', '5', str(frames_dir/'frame_%05d.jpg')]
            run(cmd, 'Extraer fotogramas')
            jpgs = sorted(frames_dir.glob('frame_*.jpg'))
            # Un fotograma siempre que el vídeo tenga contenido.
            if not jpgs:
                cmd = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                       '-i', str(clip_out), '-frames:v', '1',
                       str(frames_dir/'frame_00001.jpg')]
                run(cmd, 'Extraer fotograma inicial')
                jpgs = sorted(frames_dir.glob('frame_*.jpg'))
            times = [round(start + i*args.sample_seconds, 3) for i in range(len(jpgs))]
            for i, img in enumerate(jpgs):
                img.rename(frames_dir/f't_{int(times[i]*1000):010d}ms.jpg')
            jpgs = sorted(frames_dir.glob('t_*.jpg'))
            contact_sheets(jpgs, tmp_dir, times)

            with (tmp_dir/'intervalos_4s.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['video_original', 'parte', 'inicio_segundos',
                                                       'fin_segundos', 'etiqueta', 'comentario'])
                writer.writeheader()
                count = math.ceil(span/args.clip_seconds)
                for k in range(count):
                    a, b = start+k*args.clip_seconds, min(start+(k+1)*args.clip_seconds,length)
                    if b-a >= .25:
                        writer.writerow({'video_original':source.name, 'parte':part+1,
                                         'inicio_segundos':f'{a:.2f}', 'fin_segundos':f'{b:.2f}',
                                         'etiqueta':'', 'comentario':''})
            (tmp_dir/'metadata.json').write_text(json.dumps({
                'video_original': source.name, 'ruta_relativa_original': str(source.relative_to(input_root)),
                'id_video': name, 'parte':part+1, 'total_partes':total_parts,
                'inicio_en_original_segundos':round(start,3), 'duracion_original_segundos':round(length,3),
                'duracion_parte_segundos':round(span,3), 'fps_revision':args.fps,
                'fotogramas_cada_segundos':args.sample_seconds,
                'intervalo_etiquetado_segundos':args.clip_seconds,
                'audio_eliminado':True, 'nota':'Los tiempos del CSV se refieren al vídeo original'
            }, ensure_ascii=False, indent=2), encoding='utf-8')
            (tmp_dir/'LEEME.txt').write_text(
                'RabbitSense — material para revisión visual\n'
                'video_reducido.mp4 contiene el MOV original comprimido (sin audio).\n'
                'fotogramas/: imágenes estáticas con marcas de tiempo originales (milisegundos).\n'
                'mosaico_*.jpg: vista rápida; NO permite confirmar movimientos por sí sola.\n'
                'intervalos_4s.csv: los intervalos para clasificar mirando video_reducido.mp4.\n'
                'Los tiempos del CSV corresponden al original; revisa metadata.json para el inicio de esta parte.\n'
                'No interpreta emociones ni diagnostica estrés/dolor.\n', encoding='utf-8')
            with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_STORED) as archive:
                for path in sorted(tmp_dir.rglob('*')):
                    if path.is_file():
                        archive.write(path, path.relative_to(tmp_dir))
            size_mb = destination.stat().st_size / 1024**2
            print(f'  {destination.name}: {size_mb:.1f} MB, {len(jpgs)} imágenes, {span:.1f} s')
            if size_mb > args.warn_mb:
                print('  AVISO: ZIP grande; reduce --size/--fps o aumenta --crf para hacerlo menor.')
            written.append(destination)
    return written


def main():
    parser = argparse.ArgumentParser(description='Prepara ZIPs pequeños con vídeos y fotogramas de Cinna.')
    parser.add_argument('carpeta', type=Path, help='Carpeta con MOV/MP4/etc. Se buscan subcarpetas.')
    parser.add_argument('--out', type=Path, default=Path.home()/'Downloads'/'Cinna_Para_ChatGPT')
    parser.add_argument('--chunk-seconds', type=int, default=120, help='Duración máxima de cada ZIP (120s)')
    parser.add_argument('--sample-seconds', type=int, default=4, help='Un fotograma JPG cada N segundos')
    parser.add_argument('--clip-seconds', type=int, default=4, help='Tramos propuestos para etiquetar')
    parser.add_argument('--fps', type=int, default=12, help='FPS del vídeo pequeño (12)')
    parser.add_argument('--size', type=int, default=720, help='Lado máximo del vídeo pequeño (720)')
    parser.add_argument('--crf', type=int, default=31, help='Compresión H.264 (mayor = ZIP más pequeño)')
    parser.add_argument('--warn-mb', type=float, default=80)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if not 10<=args.chunk_seconds<=3600 or not 1<=args.sample_seconds<=60 or not 2<=args.clip_seconds<=60:
        parser.error('Revisa los intervalos de tiempo: chunk 10-3600, sample 1-60, clip 2-60.')
    if not 2<=args.fps<=30 or not 128<=args.size<=1920 or not 18<=args.crf<=42:
        parser.error('Revisa --fps (2-30), --size (128-1920), --crf (18-42).')
    root = args.carpeta.expanduser().resolve()
    if not root.is_dir():
        parser.error(f'No existe la carpeta de vídeos: {root}')
    out = args.out.expanduser().resolve()
    if out == root or root in out.parents:
        parser.error('La carpeta de salida no debe estar dentro de la carpeta de vídeos.')
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not files:
        parser.error(f'No hay vídeos compatibles en {root}')
    out.mkdir(parents=True, exist_ok=True)
    ffmpeg = get_ffmpeg()
    print('Vídeos encontrados:', len(files), '\nSalida:', out)
    errors = []
    completed = []
    for idx, p in enumerate(files, 1):
        print(f'[{idx}/{len(files)}] {p.relative_to(root)}')
        try:
            completed.extend(export_video(p, root, out, ffmpeg, args))
        except Exception as exc:
            print('  ERROR:', exc, file=sys.stderr)
            errors.append((str(p), str(exc)))
    (out/'resumen_exportacion.json').write_text(json.dumps({
        'archivos_zip': [p.name for p in completed], 'errores': errors
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nZIPs listos: {len(completed)}. Errores: {len(errors)}. Directorio: {out}')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
