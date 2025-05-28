import requests
import base64

response = requests.post("http://127.0.0.1:5000/solve", json={"k": 1.0})

if response.status_code == 200:
    result = response.json()
    if result["success"]:
        data = result["data"]
        print("The minimum height:", min(data["h"]))

        with open("trajectory.png", "wb") as f:
            f.write(base64.b64decode(data["image_base64"]))
            print("Image has saved as trajectory.png")

    else:
        print("Decoding failed", result.get("error", "Unknown error"))

else:
    print("Network request failed, status code:", response.status_code)