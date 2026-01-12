import os
import time
import re
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# -----------------------------------------
# FINAL PLAYWRIGHT SCRAPER (VISIBLE MODE)
# -----------------------------------------

TOPIC_URLS = {
    "Operational Amplifiers Op-Amps": [
        "https://www.electronics-tutorials.ws/opamp/opamp_1.html",
        "https://www.tutorialspoint.com/operational_amplifier/op_amp_introduction.htm",
        "https://www.analog.com/en/education/education-library/introductory.html"
    ],
    "MOSFET basics": [
        "https://www.electronics-tutorials.ws/transistor/tran_6.html",
        "https://www.elprocus.com/mosfet-basics-types-and-applications/"
    ],
    "Flip-Flops digital electronics": [
        "https://www.electronics-tutorials.ws/sequential/seq_1.html",
        "https://www.tutorialspoint.com/digital_circuits/digital_circuits_flip_flops.htm"
    ],
    "Counters digital circuits": [
        "https://www.electronics-tutorials.ws/counter/count_1.html",
        "https://www.tutorialspoint.com/digital_circuits/digital_circuits_counters.htm"
    ],
    "Convolution signals": [
        "https://www.tutorialspoint.com/signals_and_systems/signals_and_systems_convolution.htm"
    ],
    "Fourier Transform basics": [
        "https://www.tutorialspoint.com/fourier_transform/fourier_transform_introduction.htm",
        "https://en.wikipedia.org/wiki/Fourier_transform"
    ],
    "Amplitude Modulation AM": [
        "https://www.electronics-tutorials.ws/amplifier/am_1.html",
        "https://www.tutorialspoint.com/amplitude_modulation/index.htm"
    ],
    "Frequency Modulation FM": [
        "https://www.electronics-tutorials.ws/communication/frequency-modulation.html",
        "https://www.tutorialspoint.com/frequency_modulation/index.htm"
    ],
    "Semiconductor Physics basics": [
        "https://www.physics-and-radio-electronics.com/electronic-devices/semiconductor/what-is-semiconductor.html",
        "https://en.wikipedia.org/wiki/Semiconductor"
    ],
    "Rectifiers electronics": [
        "https://www.electronics-tutorials.ws/power/diode_6.html",
        "https://www.tutorialspoint.com/rectifiers/index.htm"
    ]
}

OUTPUT_DIR = "db/raw/"


def ensure_output_dir():
    if not os.path.exists("db"):
        os.makedirs("db")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def clean(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def scrape_with_playwright(url, page):
    print(f"[SCRAPE] Opening → {url}")

    try:
        page.goto(url, timeout=60000)  # Allow 60 sec
        time.sleep(3)  # Allow JS rendering

        html = page.content()
        soup = BeautifulSoup(html, "html.parser")

        # Remove scripts
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(" ")
        cleaned = clean(text)

        print(f"[SCRAPE] Extracted {len(cleaned)} chars from page")
        return cleaned

    except Exception as e:
        print(f"[ERROR] Problem scraping {url}: {e}")
        return ""


def scrape_topic(topic, urls):
    print("\n========================================")
    print(f"   SCRAPING TOPIC → {topic}")
    print("========================================")

    final_text = ""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Visible browser
        page = browser.new_page()

        for url in urls:
            content = scrape_with_playwright(url, page)
            final_text += " " + content

        browser.close()

    final_text = clean(final_text)

    filename = f"raw_{topic.lower().replace(' ', '_').replace('-', '_')}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"[SAVE] → {filepath} ({len(final_text)} chars)")


if __name__ == "__main__":
    ensure_output_dir()

    print("\n===== PLAYWRIGHT SCRAPER STARTED =====\n")

    for topic, urls in TOPIC_URLS.items():
        scrape_topic(topic, urls)

    print("\n===== SCRAPING COMPLETE =====\n")
