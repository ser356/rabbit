# Workspace consolidado

El código activo permanece en este directorio. RabbitSense_Training y RabbitSense_ARCHIVO son recursos hermanos; no es necesario copiarlos a Descargas ni descargar ZIP.

1. Instalar el paquete editable siguiendo [README.md](README.md).
2. Ejecutar pruebas y suite existente.
3. Si cambia la ubicación de los datos, editar [benchmarks/local.json](benchmarks/local.json), o pasar rutas explícitas al CLI.

No reutilizar un entorno virtual copiado: sus enlaces editables pueden apuntar al directorio anterior. El recibido apuntaba a Descargas y se reparó reinstalando el paquete.

Vídeos, pesos y resultados voluminosos siguen fuera de Git. El historial único está en la raíz del workspace e incluye el código recibido antes de modificarlo. Ninguna migración borra datos originales.
