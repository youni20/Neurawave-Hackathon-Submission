import requests

# REPLACE WITH YOUR UBUNTU SERVER'S IP ADDRESS
SERVER_URL = "http://100.89.109.97:5000/chat"

def main():
    print("--- Connected to Nova (Remote) ---")
    chat_history = []

    while True:
        user_input = input("\nYOU: ")
        if user_input.lower() == 'exit': break

        # Prepare data to send
        payload = {
            'message': user_input,
            'history': chat_history
        }

        try:
            # Send request to your server
            response = requests.post(SERVER_URL, json=payload)
            data = response.json()

            # Update our local history with the new history from server
            chat_history = data['history']

            # Display the character's reaction
            print(f"Dr.Neura ({data['emotion']}): {data['text']}")

        except Exception as e:
            print(f"Error connecting to server: {e}")

if __name__ == "__main__":
    main()