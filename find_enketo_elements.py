with open('kobo_page.html', 'r', encoding='utf-8') as f:
    text = f.read()

import re
print("--- Links and Scripts ---")
for m in re.finditer(r'(href|src)=["\']([^"\']+)["\']', text):
    print(m.group(1), m.group(2))

print("--- Form or Article Tags ---")
for m in re.finditer(r'<(article|form|section|main)\b[^>]*>', text):
    print(m.group(0))

print("--- Any IDs or classes ---")
for m in re.finditer(r'id=["\']([^"\']+)["\']', text):
    print("id:", m.group(1))
