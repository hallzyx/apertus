# Demo real de Apertus Evidence Lab

[Ver o descargar el video MP4](apertus-demo.mp4?raw=true).

Grabación de la pantalla de la app en Chromium: PDF oficial de 72 páginas →
afirmación → Apertus v1.5-8B con cabeza entrenada y recuperación híbrida 8k →
etiqueta, probabilidad, citas/páginas, tokens y tiempo → descarga JSON.

Se comprobó primero el recorrido con tres afirmaciones y después se repitió
completo con grabación, manteniendo el mismo backend cargado entre ambas pasadas.
La segunda pasada reprodujo las etiquetas 0, 0 y 2. Son ejemplos de funcionamiento
sin etiquetas de evaluación anotadas; no constituyen una medición de precisión
ni cambian el Macro-F1 del laboratorio. No se usó el test interno de 310 ejemplos,
no hubo entrenamiento, ni se sustituyó la cabeza por la API base de CSCS.

El MP4 dura 47.56 segundos, resolución 1440×1100, sin audio.
Es una conversión H.264 de la grabación original, sin cortes ni aceleración.
Se revisaron imágenes del inicio, espera de inferencia y resultado final.

| Afirmación | Texto | Etiqueta | Tokens | Tiempo registrado, segunda pasada |
|---|---|---:|---:|---:|
| DE | Die schweizerische Währung ist der Franken. | 0 | 8072 | 2.235 s |
| FR | Le contre-projet direct propose d’inscrire dans la Constitution que la monnaie suisse est le franc. | 0 | 8149 | 2.153 s |
| IT | Il controprogetto diretto stabilisce che la valuta svizzera è l’euro. | 2 | 8161 | 2.185 s |

0 = Entailment / apoyada; 1 = Neutral / información insuficiente;
2 = Contradiction / contradicha. Los tiempos incluyen recuperación e inferencia
con la GPU ya preparada; no son garantías para otras máquinas o documentos.
La primera consulta de la primera pasada tardó 14.079 s con preparación fría.
Las citas son texto original del PDF y acompañan la decisión; no certifican
por sí mismas evidencia mínima suficiente.

## Repetir en tu PC

Arranca la app con la [guía para PC](../PROBAR_EN_PC.md) y configura el backend
nativo. Sube [este mismo folleto oficial](../../track_2a/data/demo/votation-2026-03-08-fr.pdf)
y prueba las afirmaciones de la tabla. Sin el backend puedes cargar el PDF e
inspeccionar pasajes, pero la clasificación permanece deshabilitada.

Las GPU temporales se conservaron durante la comprobación y la grabación.
Después de recuperar y verificar el video y sus resultados, ambas instancias de
la tarea fueron eliminadas y su ausencia comprobada en el listado del proveedor.
No queda un backend persistente publicado como resultado de esta demo.

## Evidencia reproducible

- [Prueba de video e integridad](video-integrity.json).
- [Recuperación, saldo y cierre de recursos](resource-verification.json).
- [Verificación de seis inferencias reales](../../track_2a/experiments/apertus-live-demo-recorded-v1/verification.json).
- [Script del recorrido y la grabación](../../track_2a/scripts/demo_v15_browser.py).

El primer intento falló al descargar el PDF desde su servidor oficial, después
de verificar los pesos. Se conservó el diagnóstico y se corrigió usando una copia
pequeña sin modificaciones, con el SHA-256 fijado y atribución a la Cancillería
Federal. El recorrido exitoso y su repetición usaron la misma GPU, sin apagar
Apertus entre ambas. Los tokens de usuario nunca se incluyeron en archivos de Git.
