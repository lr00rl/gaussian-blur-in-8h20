# test_upload.py
import requests

# 测试单张
with open('../gaussian-blur/0000220505603.jpg', 'rb') as f:
    img_data = f.read()
    print(f"Image size: {len(img_data)} bytes")
    print(f"Magic number: {img_data[:4].hex()}")

files = [
    ('files', ('test.jpg', img_data, 'image/jpeg'))
]

response = requests.post(
    'http://localhost:9000/blur/batch',
    files=files,
    data={'quality': '75'}
)

print(f"Status: {response.status_code}")
print(f"Content-Type: {response.headers.get('content-type')}")
print(f"Response size: {len(response.content)} bytes")

if response.status_code == 200:
    with open('output.zip', 'wb') as f:
        f.write(response.content)
    print("✓ Saved to output.zip")
else:
    print(f"✗ Error: {response.text}")
