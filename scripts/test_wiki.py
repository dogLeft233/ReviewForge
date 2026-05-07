import httpx

client = httpx.Client(timeout=30.0, follow_redirects=True)
try:
    # Simulate what the retriever does: same headers, same flow
    resp = client.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": "diffusion model image generation",
            "srlimit": 5,
            "format": "json",
        },
        headers={
            "User-Agent": "ReviewForge/1.0 (https://github.com/reviewforge; mailto:research@example.com)",
            "Accept": "application/json",
        },
    )
    print("query+search status:", resp.status_code)
    print("Response:", resp.text[:200])
except Exception as e:
    print("Failed:", e)
finally:
    client.close()
