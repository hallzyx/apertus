# Revisión visual y funcional de Apertus Evidence Lab

Revisión realizada el 12 de octubre de 2026 (UTC) en Chromium 151.0.7922.173,
con Playwright y la aplicación HTTP real. La interfaz sigue siendo autónoma:
no requiere fuentes remotas, bibliotecas JavaScript ni una compilación frontend.

## Diseño aplicado

Identidad editorial inspirada en documentación cívica suiza: papel cálido,
verde bosque, sello geométrico rojo y composición tipográfica clara. El flujo
se organiza en documento, afirmación y evidencia. La interfaz admite afirmaciones
DE/FR/IT, también en un idioma distinto al documento. Los resultados presentan
pasajes con páginas e identificadores; el JSON completo queda disponible en
un desplegable y como descarga.

La pantalla empieza sin documento ni afirmación ficticios. El ejemplo sintético
requiere una acción explícita y queda identificado tanto en la entrada como en
los resultados. Editar la consulta elimina resultados anteriores. Cargar otro
PDF invalida la fuente anterior antes de enviarlo: una carga fallida no puede
reutilizar silenciosamente el ejemplo o el documento previo.

Las solicitudes bloquean temporalmente entradas y acciones para impedir estados
cruzados. Hay mensajes de progreso, errores recuperables, foco tras los resultados,
etiquetas de campos, enlace para saltar al laboratorio y tratamiento de movimiento
reducido. La configuración del backend no se presenta como conectividad verificada.

Se conserva la correspondencia científica 0/entailment, 1/neutral y
2/contradiction. Solo se muestran probabilidades devueltas por el backend,
con una aclaración sobre calibración; no se fabrican confianza, tokens, tiempos
ni citas. La interfaz avisa si la respuesta indica truncamiento.

## Capturas verificadas

- [Escritorio, estado inicial, 1440 px](desktop.png)
- [Móvil, estado inicial, 390 px](mobile.png)
- [Escritorio con pasajes, 1440 px](desktop-evidence.png)
- [Móvil con pasajes, 390 px](mobile-evidence.png)

Las capturas de pasajes usan la recuperación real y el ejemplo sintético señalado
en pantalla. No muestran inferencia GPU ni resultados de un folleto oficial.

## Validación

Las 41 pruebas existentes de `track_2a/tests` pasan. Además se añadieron seis
pruebas opcionales de navegador en `research_tests/test_web_browser.py`:

1. Recuperación HTTP real sin backend, descarga JSON e invalidación al editar.
2. Carga y recuperación reales de un PDF textual sintético, conservando su página.
3. Carga PDF pendiente/fallida: controles bloqueados, error enfocado y fuente anterior eliminada.
4. Correspondencia de las tres etiquetas, probabilidades ausentes, truncamiento y texto HTML tratado como texto.
5. Error de inferencia: controles recuperados sin emitir una decisión.
6. Ausencia de desbordamiento horizontal en 320, 390, 768, 1024 y 1440 px; también con pasajes a 320 px.

```bash
PYTHONPATH=track_2a/src python -m unittest discover -s track_2a/tests -q
PYTHONPATH=track_2a/src python -m unittest discover -s track_2a/research_tests -p test_web_browser.py -v
```

Las pruebas de navegador requieren el paquete Python `playwright` y Chromium
local; la de PDF también requiere el extra `pdf` del proyecto. Si faltan se
omiten expresamente. No añaden dependencias a la aplicación desplegada.

## Alcance

La carga de PDF y la recuperación se comprobaron con los endpoints reales.
Las respuestas de clasificación y los fallos controlados del navegador usan
fixtures sintéticos. La revisión no valida inferencia GPU, calidad predictiva,
Safari/Firefox ni el uso con lectores de pantalla. Los recursos de modelo,
datasets y política de inferencia no se modificaron.

## Selector de idiomas

La interfaz incorpora ES/EN/DE/FR/IT, detección inicial del navegador y preferencia
persistente. Se comprobaron `es-PE`, `de-CH`, `fr-CH`, `it-CH`, `rm-CH`, `ja-JP`
y `de-DE`; los dos últimos usan inglés. `rm-CH` usa alemán como alternativa.
Los tres nuevos tests de Chromium comprueban detección/persistencia, cambios con
una consulta y citas abiertas sin alterar el texto de origen, y mensajes en inglés
sin desbordamiento a 320 px. En total son nueve tests de navegador.

Capturas adicionales: [inglés](english-desktop.png), [alemán móvil](german-mobile.png),
[francés](french-desktop.png) e [italiano móvil](italian-mobile.png).
