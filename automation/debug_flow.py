from playwright.sync_api import sync_playwright
import time
import json
from pathlib import Path

def inspect_flow():
    with sync_playwright() as pw:
        context = pw.chromium.launch_persistent_context(
            user_data_dir=r"C:\FlowBotProfile",
            channel="chrome",
            headless=False,
            args=["--start-maximized"]
        )
        page = context.pages[0] if context.pages else context.new_page()
        
        print("Navigating to Flow...")
        page.goto("https://flow.google.com/u/4/project/2a0986c0-3556-41b6-956e-d45046786417")
        time.sleep(5)
        
        print("Opening settings...")
        page.locator('button[aria-label="Settings trigger"]').click()
        time.sleep(2)
        
        popover = page.locator('.cdk-overlay-pane').last
        print("Taking settings screenshot...")
        page.screenshot(path="settings_panel.png")
        
        print("Dumping settings DOM...")
        dom_html = popover.evaluate("el => el.outerHTML")
        Path("settings_dom_debug.html").write_text(dom_html, encoding="utf-8")
        
        print("Done!")

if __name__ == "__main__":
    inspect_flow()
