import requests

lon = float(input("Enter Longitude: "))
lat = float(input("Enter Latitude: "))
url = f"https://opendata-download-metfcst.smhi.se/api/category/pmp3g/version/2/geotype/point/lon/{lon}/lat/{lat}/data.json"

response = requests.get(url)

print(response)