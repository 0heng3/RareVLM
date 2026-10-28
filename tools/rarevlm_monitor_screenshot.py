"""Capture the local dashboard using an isolated optional Playwright install."""
import json
import os
from pathlib import Path
import sys
if os.environ.get('RAREVLM_MONITOR_RENDERER', 'offline') == 'offline':
    from rarevlm_monitor_raster import render
    result = render(json.loads((Path(sys.argv[2]) / 'latest.json').read_text()), Path(sys.argv[2]))
    (Path(sys.argv[2]) / 'RENDER_CHECK.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False)); sys.exit(0)
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    from rarevlm_monitor_raster import render
    result = render(json.loads((Path(sys.argv[2]) / 'latest.json').read_text()), Path(sys.argv[2]))
    (Path(sys.argv[2]) / 'RENDER_CHECK.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False)); sys.exit(0)

page_path, output = map(Path, sys.argv[1:3]); output.mkdir(parents=True, exist_ok=True)
font_dir = Path('/opt/data/private/.tools/fonts')
config = output / 'fontconfig.conf'
config.write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd"><fontconfig><dir>'+str(font_dir)+'</dir><cachedir>'+str(output / 'cache/fontconfig')+'</cachedir></fontconfig>')
os.environ['FONTCONFIG_FILE'] = str(config)
errors = []
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
    page = browser.new_page(viewport={'width': 1440, 'height': 1150}, device_scale_factor=1)
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(page_path.resolve().as_uri() + '?capture=1', wait_until='load')
    page.wait_for_function('window.__MONITOR_READY__ === true'); page.evaluate('document.fonts.ready'); page.screenshot(path=str(output / '总览.png'), full_page=True)
    page.get_by_role('button', name='错误与案例', exact=True).click(); page.screenshot(path=str(output / '错误原因.png'), full_page=True)
    page.get_by_role('button', name='训练与评测', exact=True).click(); page.screenshot(path=str(output / '训练曲线.png'), full_page=True)
    # Exercise actual filters and modal; no experiment mutations or extra scoring.
    page.get_by_role('button', name='错误与案例', exact=True).click(); page.locator('#case-filter').select_option('rejected')
    page.locator('#case-search').fill('Apple'); matches = page.locator('#cases article').count()
    if not matches: raise AssertionError('Expected Apple equivalence case in actual review')
    page.locator('#cases img').first.click(); page.locator('#close-modal').click()
    page.locator('#case-search').fill(''); page.locator('#case-filter').select_option('loss')
    report = {'status': 'passed' if not errors else 'failed', 'browser_errors': errors, 'apple_filter_cards': matches,
              'rendered_loss_cases': page.locator('#cases article').count(), 'overview_png': str(output / '总览.png')}
    browser.close()
Path(output / 'BROWSER_CHECK.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False))
if errors: sys.exit(1)
