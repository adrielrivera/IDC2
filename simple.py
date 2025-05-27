import serial
import time
import random

# Configure serial connection with both read and write timeouts
ser = serial.Serial(
    port='/dev/ttyUSB0',
    baudrate=9600,
    timeout=1,
    write_timeout=1
)
time.sleep(2)

print("Raspberry Pi detection server running...")

while True:
    if ser.in_waiting > 0:
        command = ser.readline().decode('utf-8').strip()
        print(f"Received command: {command}")  # Debug print
        
        if command == "REQUEST_DETECTION":
            print("Detection requested")
            
            # Simulate detection
            detected = random.choice([True, False])
            
            try:
                if detected:
                    response = "DETECTED:hotdog:0.95\n"
                    print(f"Sending: {response.strip()}")  # Debug print
                    ser.write(response.encode('utf-8'))
                    ser.flush()  # Ensure data is sent
                else:
                    response = "NO_DETECTION\n"
                    print(f"Sending: {response.strip()}")  # Debug print
                    ser.write(response.encode('utf-8'))
                    ser.flush()  # Ensure data is sent
                
            except Exception as e:
                print(f"Error sending response: {e}")
                
    time.sleep(0.1)