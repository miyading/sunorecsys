import requests
from bs4 import BeautifulSoup
import json

# Step 1: get the page HTML
url = "https://suno.com/explore"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
resp = requests.get(url, headers=headers)
html = resp.text

# Step 2: parse HTML
soup = BeautifulSoup(html, "html.parser")

# Step 3: find all divs with class 'genreItem'
genres = [div.get_text(strip=True) for div in soup.find_all("div", class_="genreItem")]

# Step 4: remove duplicates
genres = sorted(set(genres))

# Step 5: save
with open("suno_genres.json", "w") as f:
    json.dump(genres, f, indent=2, ensure_ascii=False)

print(f"✅ Extracted {len(genres)} genres:")
print(genres[:20])  # preview first 20
