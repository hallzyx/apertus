"""Optional real Chromium UI checks. Run with PYTHONPATH=track_2a/src python
-m unittest discover -s track_2a/research_tests -p test_web_browser.py -v.
Requires Playwright and Chromium; all model responses here are synthetic fixtures.
"""
import json
import importlib.util
import shutil
import threading
import unittest
from http.server import ThreadingHTTPServer

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

from ost_nli.web import Handler


@unittest.skipUnless(sync_playwright and shutil.which('chromium'), 'Requires Playwright and Chromium')
class WebBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(
            executable_path=shutil.which('chromium'), headless=True, args=['--no-sandbox'])

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        self.page = self.browser.new_page(locale='es-ES', viewport={'width': 1440, 'height': 1000})
        self.errors = []
        self.page.on('pageerror', lambda error: self.errors.append(str(error)))

    def tearDown(self):
        self.page.close()
        self.assertEqual(self.errors, [])

    def open(self, configured=False):
        self.page.route('**/health', lambda route: route.fulfill(
            json={'model_configured': configured, 'model_connectivity_verified': False}))
        self.page.goto(f'http://127.0.0.1:{self.server.server_port}', wait_until='networkidle')

    def demo(self):
        self.page.locator('.advanced > summary').click()
        self.page.locator('#demo').click()

    def test_offline_retrieval_is_real_and_edits_clear_old_results(self):
        self.open()
        self.assertTrue(self.page.locator('#predict').is_disabled())
        self.assertTrue(self.page.locator('#retrieve').is_disabled())
        self.demo()
        self.assertFalse(self.page.locator('#retrieve').is_disabled())
        self.page.locator('#retrieve').click()
        self.page.locator('.passage').first.wait_for()
        self.assertIn('SINTÉTICO', self.page.locator('#summary').inner_text())
        self.assertIn('PÁGINA', self.page.locator('.passage').first.inner_text())
        self.assertTrue(self.page.locator('#predict').is_disabled())
        with self.page.expect_download() as download:
            self.page.locator('#download').click()
        with open(download.value.path()) as file:
            self.assertGreater(len(json.load(file)['evidence']), 0)
        self.page.locator('#claim').fill('Eine andere Aussage.')
        self.assertTrue(self.page.locator('#download').is_hidden())
        self.assertEqual(self.page.locator('.passage').count(), 0)

    @unittest.skipUnless(importlib.util.find_spec('pypdf'), 'Requires PDF extra')
    def test_text_pdf_upload_and_real_retrieval_preserve_page(self):
        # A tiny synthetic PDF generated in memory, never an official booklet.
        stream = b'BT /F1 12 Tf 40 700 Td (Die Vorlage kostet 100 Franken.) Tj ET'
        objects = [b'<< /Type /Catalog /Pages 2 0 R >>',
                   b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
                   b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
                   b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
                   b'<< /Length '+str(len(stream)).encode()+b' >>\nstream\n'+stream+b'\nendstream']
        pdf = b'%PDF-1.4\n'
        offsets = []
        for index, value in enumerate(objects, 1):
            offsets.append(len(pdf))
            pdf += str(index).encode()+b' 0 obj\n'+value+b'\nendobj\n'
        xref = len(pdf)
        pdf += b'xref\n0 6\n0000000000 65535 f \n'
        pdf += b''.join(f'{offset:010d} 00000 n \n'.encode() for offset in offsets)
        pdf += b'trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n'+str(xref).encode()+b'\n%%EOF'
        self.open()
        self.page.locator('#pdf').set_input_files(
            {'name': 'synthetic-test.pdf', 'mimeType': 'application/pdf', 'buffer': pdf})
        self.page.wait_for_function("document.getElementById('source').textContent.includes('1 página')")
        self.page.locator('#claim').fill('Die Vorlage kostet 100 Franken.')
        self.page.locator('#retrieve').click()
        self.page.locator('.passage').first.wait_for()
        self.assertIn('PÁGINA 1', self.page.locator('.passage').first.inner_text())
        self.assertIn('100 Franken', self.page.locator('blockquote').first.inner_text())

    def test_pending_and_failed_pdf_cannot_reuse_previous_document(self):
        self.open(configured=True)
        self.demo()
        pending = []
        self.page.route('**/api/document', lambda route: pending.append(route))
        self.page.locator('#pdf').set_input_files(
            {'name': 'unreadable.pdf', 'mimeType': 'application/pdf', 'buffer': b'%PDF-invalid'})
        self.page.wait_for_function("document.getElementById('results').getAttribute('aria-busy') === 'true'")
        for selector in ['#retrieve', '#predict', '#claim', '#pdf', '#doc', '#demo']:
            self.assertTrue(self.page.locator(selector).is_disabled(), selector)
        self.assertEqual(self.page.locator('#doc').input_value(), '')
        self.page.wait_for_timeout(100)
        self.assertEqual(len(pending), 1)
        pending[0].fulfill(status=400, json={'error': 'Synthetic PDF extraction failure'})
        self.page.wait_for_function("document.getElementById('feedback').dataset.kind === 'error'")
        self.assertTrue(self.page.locator('#retrieve').is_disabled())
        self.assertTrue(self.page.locator('#predict').is_disabled())
        self.assertEqual(self.page.locator('#pdf').input_value(), '')
        self.assertFalse(self.page.locator('#claim').is_disabled())
        self.assertTrue(self.page.locator('#feedback').evaluate('(node) => node === document.activeElement'))

    def test_label_mapping_and_unavailable_probabilities_are_honest(self):
        self.open(configured=True)
        self.demo()
        labels = ['Apoyada por el documento', 'Información insuficiente', 'Contradicha por el documento']
        for label, title in enumerate(labels):
            with self.subTest(label=label):
                self.page.route('**/api/predict', lambda route, request, label=label: route.fulfill(json={
                    'label': label, 'input_tokens': 42, 'inference_time_ms': 123,
                    'evidence': [{'id': 'synthetic-p1', 'page': 2, 'text': '<script>unsafe()</script> synthetic fixture'}],
                    'truncated': True}))
                self.page.locator('#predict').click()
                self.page.wait_for_function("document.getElementById('result-state').textContent === 'EVALUACIÓN'")
                self.assertEqual(self.page.locator('.result-title').inner_text(), title)
                self.assertIn('No disponible', self.page.locator('.metrics').inner_text())
                self.assertIn('Contexto truncado', self.page.locator('.warning').inner_text())
                self.assertIn('<script>', self.page.locator('blockquote').inner_text())
                self.assertEqual(self.page.locator('blockquote script').count(), 0)
                self.page.unroute('**/api/predict')

    def test_failed_prediction_recovers_controls_without_result(self):
        self.open(configured=True)
        self.demo()
        self.page.route('**/api/predict', lambda route: route.fulfill(status=502, json={'error': 'Unavailable'}))
        self.page.locator('#predict').click()
        self.page.wait_for_function("document.getElementById('feedback').dataset.kind === 'error'")
        self.assertIn('ninguna decisión', self.page.locator('#feedback').inner_text())
        self.assertFalse(self.page.locator('#predict').is_disabled())
        self.assertTrue(self.page.locator('#download').is_hidden())
        self.assertEqual(self.page.locator('.result-title').count(), 0)

    def test_mobile_tablet_and_desktop_have_no_horizontal_overflow(self):
        self.open()
        for width in [320, 390, 768, 1024, 1440]:
            with self.subTest(width=width):
                self.page.set_viewport_size({'width': width, 'height': 900})
                self.assertFalse(self.page.evaluate('document.documentElement.scrollWidth > window.innerWidth'))
        self.demo()
        self.page.locator('#retrieve').click()
        self.page.locator('.passage').first.wait_for()
        self.page.set_viewport_size({'width': 320, 'height': 900})
        self.assertFalse(self.page.evaluate('document.documentElement.scrollWidth > window.innerWidth'))

    def test_language_detection_and_saved_override(self):
        for locale, expected in [('es-PE','es'),('de-CH','de'),('fr-CH','fr'),('it-CH','it'),('rm-CH','de'),('ja-JP','en'),('de-DE','en')]:
            with self.subTest(locale=locale):
                context=self.browser.new_context(locale=locale)
                page=context.new_page()
                page.goto(f'http://127.0.0.1:{self.server.server_port}',wait_until='networkidle')
                self.assertEqual(page.locator('html').get_attribute('lang'),expected)
                self.assertEqual(page.locator('#ui-language').input_value(),expected)
                page.select_option('#ui-language','es')
                page.reload(wait_until='networkidle')
                self.assertEqual(page.locator('html').get_attribute('lang'),'es')
                context.close()

    def test_language_switch_preserves_query_and_source_text(self):
        self.open()
        self.demo()
        claim=self.page.locator('#claim').input_value()
        document=self.page.locator('#doc').input_value()
        self.page.locator('#retrieve').click()
        self.page.locator('.passage').first.wait_for()
        quote=self.page.locator('blockquote').first.inner_text()
        for language in ['en','de','fr','it','es']:
            self.page.select_option('#ui-language',language)
            self.assertEqual(self.page.locator('html').get_attribute('lang'),language)
            self.assertEqual(self.page.locator('#claim').input_value(),claim)
            self.assertEqual(self.page.locator('#doc').input_value(),document)
            self.assertEqual(self.page.locator('blockquote').first.inner_text(),quote)
            self.assertFalse(self.page.locator('#download').is_hidden())
        self.assertIn('Seleccionar PDF',self.page.locator('.file-picker').inner_text())

    def test_english_mobile_layout_and_live_messages(self):
        self.open()
        self.page.select_option('#ui-language','en')
        self.page.set_viewport_size({'width':320,'height':900})
        self.assertIn('PDF',self.page.locator('.file-picker').inner_text())
        self.assertNotIn('Seleccionar',self.page.locator('.file-picker').inner_text())
        self.assertEqual(self.page.locator('#retrieve').inner_text(),'Inspect passages')
        self.demo()
        self.assertNotIn('La fuente es',self.page.locator('#summary').inner_text())
        self.page.locator('#retrieve').click()
        self.page.locator('.passage').first.wait_for()
        self.page.wait_for_function("!document.getElementById('feedback').textContent.includes('Pasajes recuperados')")
        self.assertLessEqual(self.page.evaluate('document.documentElement.scrollWidth'),320)
