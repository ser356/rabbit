#!/bin/bash
# Opción robusta frente a la limitación de Google Drive: usa OAuth con rclone.
set -Eeuo pipefail
cd "$(dirname "$0")"
if ! command -v rclone >/dev/null 2>&1; then
  if ! command -v brew >/dev/null 2>&1; then
    echo 'Necesitas Homebrew para instalar rclone: https://brew.sh/' >&2
    exit 1
  fi
  brew install rclone
fi
printf '\nEsta opción usa tu propia autorización Google Drive mediante OAuth.\n'
printf 'No copies ni compartas archivos de cookies, tokens o contraseñas.\n'
printf 'En rclone config, crea un remoto de tipo "drive" (Google Drive), con acceso de solo lectura.\n'
printf 'Se recomienda configurar un Google OAuth client_id propio según la documentación de rclone.\n'
read -r -p '¿Quieres abrir la configuración de rclone? [s/N] ' ans
if [[ "$ans" =~ ^[sSyY]$ ]]; then rclone config; fi
rclone listremotes
read -r -p 'Nombre del remoto de Google Drive (p. ej., rabbitsa): ' remote
remote="${remote%:}"
[[ -n "$remote" ]] || { echo 'No se indicó remoto.' >&2; exit 1; }
if ! rclone listremotes | grep -Fxq "${remote}:"; then
  echo "El remoto ${remote} no existe; primero hay que crearlo con rclone config." >&2
  exit 1
fi
export RABBITSENSE_DRIVE_REMOTE="$remote"
echo 'Intentando descargar por la API de Google Drive con autorización...' 
bash start.command --download-method rclone
