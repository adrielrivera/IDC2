import serial
import time
import random  # For testing only

# Configure serial connection
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)  # Adjust port if needed
time.sleep(2)  # Wait for connection to establish

print("Raspberry Pi detection server running...")

while True:
    if ser.in_waiting > 0:
        command = ser.readline().decode('utf-8').strip()
        
        if command == "REQUEST_DETECTION":
            print("Detection requested")
            
            # Simulate detection (replace with actual model call)
            detected = random.choice([True, False])
            
            if detected:
                # Format: "DETECTED:object_name:confidence"
                ser.write(b"DETECTED:hotdog:0.95\n")
            else:
                ser.write(b"NO_DETECTION\n")
                
    time.sleep(0.1)