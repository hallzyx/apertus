# Visual and functional review of Apertus Evidence Lab

Review carried out on 12 October 2026 (UTC) in Chromium 151.0.7922.173,
with Playwright and the real HTTP application. The interface remains self-contained:
it requires no remote fonts, JavaScript libraries or frontend build.

## Applied design

Editorial identity inspired by Swiss civic documentation: warm paper, forest
green, a red geometric seal and clear typographic composition. The flow is
organized into document, claim and evidence. The interface accepts DE/FR/IT
claims, including in a language different from the document. Results present
passages with pages and identifiers; the full JSON remains available in
a disclosure panel and as a download.

The screen starts without a fictitious document or claim. The synthetic example
requires an explicit action and is identified both in the input and in the
results. Editing the query removes previous results. Loading another
PDF invalidates the previous source before sending it: a failed upload cannot
silently reuse the example or the previous document.

Requests temporarily lock inputs and actions to prevent crossed states.
There are progress messages, recoverable errors, focus after the results,
field labels, a link to skip to the lab and reduced-motion handling. The backend
configuration is not presented as verified connectivity.

The scientific mapping 0/entailment, 1/neutral and 2/contradiction is preserved.
Only probabilities returned by the backend are shown,
with a note on calibration; no confidence, tokens, times or quotes are
fabricated. The interface warns if the response indicates truncation.

## Verified screenshots

- [Desktop, initial state, 1440 px](desktop.png)
- [Mobile, initial state, 390 px](mobile.png)
- [Desktop with passages, 1440 px](desktop-evidence.png)
- [Mobile with passages, 390 px](mobile-evidence.png)

The passage screenshots use real retrieval and the synthetic example flagged
on screen. They show neither GPU inference nor results from an official booklet.

## Validation

The 41 existing tests in `track_2a/tests` pass. In addition, six optional
browser tests were added in `research_tests/test_web_browser.py`:

1. Real HTTP retrieval without a backend, JSON download and invalidation on edit.
2. Real upload and retrieval of a synthetic text PDF, preserving its page.
3. Pending/failed PDF upload: locked controls, focused error and previous source removed.
4. Mapping of the three labels, absent probabilities, truncation and HTML text treated as text.
5. Inference error: controls recovered without emitting a decision.
6. No horizontal overflow at 320, 390, 768, 1024 and 1440 px; also with passages at 320 px.

```bash
PYTHONPATH=track_2a/src python -m unittest discover -s track_2a/tests -q
PYTHONPATH=track_2a/src python -m unittest discover -s track_2a/research_tests -p test_web_browser.py -v
```

The browser tests require the Python package `playwright` and a local
Chromium; the PDF one also requires the project's `pdf` extra. If they are missing they
are explicitly skipped. They add no dependencies to the deployed application.

## Scope

PDF upload and retrieval were checked against the real endpoints.
Classification responses and controlled browser failures use synthetic
fixtures. The review does not validate GPU inference, predictive quality,
Safari/Firefox or use with screen readers. Model resources,
datasets and inference policy were not modified.

## Language selector

The interface includes ES/EN/DE/FR/IT, initial browser detection and a persistent
preference. `es-PE`, `de-CH`, `fr-CH`, `it-CH`, `rm-CH`, `ja-JP`
and `de-DE` were checked; the last two use English. `rm-CH` falls back to German.
The three new Chromium tests check detection/persistence, changes with
a query and open quotes without altering the source text, and English messages
without overflow at 320 px. In total there are nine browser tests.

Additional screenshots: [English](english-desktop.png), [German mobile](german-mobile.png),
[French](french-desktop.png) and [Italian mobile](italian-mobile.png).
