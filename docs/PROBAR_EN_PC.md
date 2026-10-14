# Probar Apertus Evidence Lab en tu PC

Necesitas Git y Docker con Compose. En Windows o macOS puedes usar Docker Desktop.
La interfaz funciona sin una GPU: extrae un PDF y muestra pasajes. Para clasificar
con el sistema entrenado necesitas además un backend nativo Apertus en GPU.

## Arrancar la app

```bash
git clone https://github.com/hallzyx/apertus.git
cd apertus
```

Copia `.env.example` a `.env` (con el explorador de archivos o `cp .env.example .env`).
Después:

```bash
docker compose up --build -d
```

Abre **http://localhost:8000** en tu navegador. Sube un folleto PDF con texto
seleccionable, escribe una afirmación DE/FR/IT y pulsa «Evaluar afirmación».
El idioma del documento puede ser diferente del de la afirmación.
Si no has configurado el backend, puedes inspeccionar el PDF y sus pasajes;
la app no genera etiquetas ni confianza simuladas.

## Conectar el sistema entrenado

El backend usa Apertus v1.5-8B y `track_2a/deployment/head-v15.json`, con recuperación
híbrida 8k. Los pesos, el modelo de recuperación y el servidor se preparan siguiendo
[la guía del backend](../track_2a/README.md#native-backend-inside-docker).
Se verificó en GPU de 48 GB; eso describe nuestro equipo probado, no una exigencia
de los organizadores. No se incluyen pesos ni tokens en Git.

Si el backend corre en el mismo PC, pon en `.env`:

```env
FROZEN_BASE_URL=http://host.docker.internal:8001
```

El servidor nativo escucha inicialmente en loopback. En Linux, para acceder desde
el contenedor, puedes usar el modo de red host descrito en `track_2a/README.md`.
En Windows/macOS, confirma que el backend sea accesible desde Docker; un backend
Linux remoto también es una opción. No expongas el servicio públicamente sin
control de acceso. `FROZEN_BASE_URL` debe apuntar a nuestro servicio con `/nli`,
no al endpoint de chat de CSCS.

Aplica cambios de configuración con:

```bash
docker compose up -d --force-recreate
```

La pantalla distingue «configurado» de «conectado»: la prueba real es una evaluación
correcta que devuelve etiqueta, pasajes, tokens y tiempo. El 0,949 es Macro-F1 en
validación interna; no certifica cada resultado ni es una puntuación de evaluación oculta.
Las citas son contexto del modelo, no una garantía automática de evidencia suficiente.

## Detener y diagnosticar

```bash
docker compose logs --tail=100 app
docker compose down
```

Si el puerto está ocupado, cambia `APP_PORT=8001` en `.env` y abre ese puerto.
Los documentos se procesan en memoria/archivos temporales y no se guardan como
una biblioteca persistente. Al clasificar se envía su texto al backend configurado.
PDF escaneado sin texto: requiere OCR previo. Puedes descargar el resultado JSON
desde la interfaz. También hay CLI documentado en `track_2a/README.md`.

## Verificación de esta entrega

Se ejecutaron las 41 pruebas existentes y se arrancó la imagen con Docker Compose.
Se cargó un PDF real de 72 páginas, se extrajeron 157 pasajes y se comprobó la
recuperación. Sin backend la clasificación permanece deshabilitada. Las pruebas
reales de Apertus del laboratorio están archivadas en el informe técnico; esta
entrega de interfaz no reevalúa el modelo ni modifica sus pesos o cabeza.

## Idioma de la interfaz

El selector del encabezado ofrece español, inglés, alemán, francés e italiano.
En la primera visita se utiliza la preferencia principal del navegador:
`es`/`es-*` → español; `de-CH`, `fr-CH`, `it-CH` → el idioma suizo
correspondiente; `rm-CH` → alemán (no hay traducción al romanche); cualquier
otro idioma → inglés. No se detecta la ubicación ni se accede a ajustes privados
del PC: se usa el idioma que el navegador informa. La selección manual queda
guardada en ese navegador. Las citas, afirmaciones y el JSON original no se traducen.

## Demo grabada con el modelo real

Puedes ver el [video y recorrido](demo/README.md). El [PDF oficial de ejemplo](../track_2a/data/demo/votation-2026-03-08-fr.pdf) está incluido para probar el mismo flujo. Para clasificar hay que conectar el backend nativo; la grabación no publica un servidor GPU permanente.
