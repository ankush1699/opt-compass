# Manually saved pages

uscis.gov and ice.gov block automated downloads (HTTP 403), and this project does not try to
get around that. To index those pages, save them yourself in a browser and drop a plain-text
file here:

- first line: `Source: <the page URL>`
- optional second line: `Title: <page title>`
- the rest: the page text

Then run `python -m app.ingest` again.
