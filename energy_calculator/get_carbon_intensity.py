import requests
import time

url = "https://api-access.electricitymaps.com/free-tier/carbon-intensity/latest?lat=49.43&lon=7.77&zone=DE"
headers = {
    "auth-token": "beupWDmvL1DJkKbmjcbnQEGDEqWILClf"
    }

def get_carbon_intensity():
    country = 'DE'
    carbon_intensity = 138
    max_retries = 2
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            data = response.json()
            country = data.get("zone")
            carbon_intensity = data.get("carbonIntensity", 138)
            return country, carbon_intensity
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except requests.exceptions.RequestException as req_err:
            print(f"Request error occurred: {req_err}")
        except Exception as e:
            print(f"Error getting carbon intensity: {e}")
        retries += 1
        delay = 2 ** retries
        print(f"Retrying in {delay} seconds...")
        time.sleep(delay)
    print(f"Unable to access geographical location. Using 'Germany' as the default value - url={url}")
    return country, carbon_intensity

def main():
    country, carbon_intensity = get_carbon_intensity()
    print(F"Country : {country}, CI : {carbon_intensity}")

if __name__ == '__main__':
    main()