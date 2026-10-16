# Try Apertus Evidence Lab on your PC

You need Git and Docker with Compose. On Windows or macOS you can use Docker Desktop.
The interface works without a GPU: it extracts a PDF and shows its passages. To classify
with the trained system you also need a native Apertus backend on a GPU.

## Start the app

```bash
git clone https://github.com/hallzyx/apertus.git
cd apertus
```

Copy `.env.example` to `.env` (with the file explorer or `cp .env.example .env`).
Then:

```bash
docker compose up --build -d
```

Open **http://localhost:8000** in your browser. Upload a booklet PDF with selectable
text, write a DE/FR/IT claim and press "Evaluate claim".
The document language can differ from the claim language.
If you have not configured the backend, you can inspect the PDF and its passages;
the app does not generate simulated labels or confidence.

## Connect the trained system

The backend uses Apertus v1.5-8B and `track_2a/deployment/head-v15.json`, with hybrid
8k retrieval. The weights, the retrieval model and the server are prepared by following
[the backend guide](../track_2a/README.md#native-backend-inside-docker).
It was verified on a 48 GB GPU; that describes our tested hardware, not a requirement
from the organizers. No weights or tokens are included in Git.

If the backend runs on the same PC, set this in `.env`:

```env
FROZEN_BASE_URL=http://host.docker.internal:8001
```

The native server initially listens on loopback. On Linux, to reach it from the
container, you can use the host network mode described in `track_2a/README.md`.
On Windows/macOS, confirm that the backend is reachable from Docker; a remote
Linux backend is also an option. Do not expose the service publicly without
access control. `FROZEN_BASE_URL` must point to our service with `/nli`,
not to the CSCS chat endpoint.

Apply configuration changes with:

```bash
docker compose up -d --force-recreate
```

The screen distinguishes "configured" from "connected": the real test is a successful
evaluation that returns a label, passages, tokens and time. The 0.949 is Macro-F1 on
internal validation; it does not certify each result and it is not a hidden-test score.
Quotes are model context, not an automatic guarantee of sufficient evidence.

## Stop and diagnose

```bash
docker compose logs --tail=100 app
docker compose down
```

If the port is busy, change `APP_PORT=8001` in `.env` and open that port.
Documents are processed in memory/temporary files and are not stored as a
persistent library. When classifying, their text is sent to the configured backend.
Scanned PDFs without text need OCR beforehand. You can download the JSON result
from the interface. There is also a CLI documented in `track_2a/README.md`.

## Verification of this delivery

The 41 existing tests were run and the image was started with Docker Compose.
A real 72-page PDF was loaded, 157 passages were extracted and retrieval was
checked. Without a backend, classification remains disabled. The real Apertus
tests of the lab are archived in the technical report; this interface delivery
does not re-evaluate the model or modify its weights or head.

## Interface language

The header selector offers Spanish, English, German, French and Italian.
On the first visit the browser's primary preference is used:
`es`/`es-*` → Spanish; `de-CH`, `fr-CH`, `it-CH` → the corresponding Swiss
language; `rm-CH` → German (there is no Romansh translation); any
other language → English. Location is not detected and private PC settings are
not accessed: the language the browser reports is used. A manual selection is
stored in that browser. Quotes, claims and the original JSON are not translated.

## Recorded demo with the real model

You can watch the [English video](demo/README.md) and the [step-by-step walkthrough with screenshots](demo/STEP_BY_STEP.md). The [official sample PDF](../track_2a/data/demo/votation-2026-03-08-fr.pdf) is included so you can try the same flow. To classify you must connect the native backend; the recording does not publish a permanent GPU server.
