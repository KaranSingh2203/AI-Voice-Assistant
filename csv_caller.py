"""
csv_caller.py
Usage: python csv_caller.py contacts.csv
CSV should have a header and column 'phone' (E.164 format like +919999999999)
It will check do_not_call.txt and skip numbers in DNC.
"""

import csv
import sys
import time
from twilio.rest import Client as TwilioRestClient
from dotenv import load_dotenv
import os

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
APP_BASE_URL = os.getenv("APP_BASE_URL")

twilio_client = TwilioRestClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

DNC_FILE = "do_not_call.txt"
if not os.path.exists(DNC_FILE):
    open(DNC_FILE, "w").close()

def load_dnc():
    with open(DNC_FILE, "r") as f:
        return set([line.strip() for line in f if line.strip()])

def place_call(to_number):
    try:
        call = twilio_client.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{APP_BASE_URL}/voice",
        )
        print("Call placed:", call.sid, to_number)
    except Exception as e:
        print("Failed to place call to", to_number, e)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python csv_caller.py contacts.csv")
        sys.exit(1)
    csv_path = sys.argv[1]
    dnc = load_dnc()
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            phone = row.get("phone") or row.get("number") or row.get("Phone")
            if not phone:
                continue
            phone = phone.strip()
            if phone in dnc:
                print("Skipping DNC:", phone)
                continue
            place_call(phone)
            time.sleep(1.0)  # avoid quick bursts; adjust
