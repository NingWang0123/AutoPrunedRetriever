import requests, os, json
r = requests.post(
    "https://api.openai.com/v1/responses",
    headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
             "Content-Type": "application/json"},
    json={"model": "gpt-4o-mini", "input": "ping"},
)
hdrs = r.headers
print("remaining requests:", hdrs.get("x-ratelimit-remaining-requests"))
print("reset requests in:", hdrs.get("x-ratelimit-reset-requests"))
print("remaining tokens:", hdrs.get("x-ratelimit-remaining-tokens"))
