"""
FinGuard - Capture High-Resolution Dashboard Screenshots for Portfolio Showcase
"""

import os
import time
from playwright.sync_api import sync_playwright

SCREENSHOTS_DIR = os.path.join(os.getcwd(), "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not os.path.exists(CHROME_PATH):
    CHROME_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

PAGES = [
    ("Executive Overview", "01_executive_overview.png"),
    ("Transaction Monitoring", "02_transaction_monitoring.png"),
    ("Anomaly Detection", "03_anomaly_detection.png"),
    ("Customer Analytics", "04_customer_analytics.png"),
    ("Model Performance", "05_model_performance.png"),
    ("SQL Insights & Reports", "06_sql_insights.png")
]

def capture_all():
    print(f"Connecting to Chrome at {CHROME_PATH}...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME_PATH,
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        
        print("Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=60000)
        time.sleep(4)  # Allow Streamlit initial hydration
        
        for nav_name, filename in PAGES:
            print(f"Navigating to page: '{nav_name}'...")
            
            # Click radio option in sidebar
            # Streamlit radio inputs have labels with the option text
            option = page.locator(f"label:has-text('{nav_name}')").first
            if option.is_visible():
                option.click()
                time.sleep(3) # Wait for page rerender & Plotly charts
            else:
                print(f"Warning: Label for '{nav_name}' not found, taking screenshot of current view.")
                
            out_path = os.path.join(SCREENSHOTS_DIR, filename)
            page.screenshot(path=out_path, full_page=False)
            print(f"Saved: {out_path} ({os.path.getsize(out_path):,} bytes)")
            
        browser.close()
        print("\nAll 6 institutional dashboard screenshots successfully captured!")

if __name__ == "__main__":
    capture_all()

