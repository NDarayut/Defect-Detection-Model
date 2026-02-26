import cv2
import serial
import time
import os

# --- CONFIGURATION ---
SERIAL_PORT = 'COM5'  # Ensure this matches your Arduino
BAUD_RATE = 9600
SAVE_FOLDER = "test_images"

def main():
    # 1. Setup Camera
    # Try 0, 1, or 2 if the camera doesn't show up
    cap = cv2.VideoCapture(1) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open camera. Try changing VideoCapture(1) to (0).")
        return

    # 2. Setup Folder
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    # 3. Setup Serial Connection
    print(f"Connecting to {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # CRITICAL: Wait for Arduino to reboot after connection
        print(f"✅ Connected to {SERIAL_PORT}")
    except Exception as e:
        print(f"❌ Serial error: {e}")
        print("Tip: Close the Arduino Serial Monitor in the IDE!")
        return

    print("\n" + "="*50)
    print("INSTRUCTIONS:")
    print("1. Click on the 'Robot View' camera window.")
    print("2. Press 's' on your keyboard to START the Arduino.")
    print("3. Press 'q' to Quit.")
    print("="*50 + "\n")
    
    img_counter = 0

    try:
        while True:
            # --- A. READ FROM ARDUINO ---
            if ser.in_waiting > 0:
                try:
                    # Read line, decode, and strip whitespace
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line: # Only print if not empty
                        print(f"🤖 Arduino says: [{line}]")

                    # IF ARDUINO SAYS IT'S READY:
                    if line == "READY_FOR_CAM":
                        print("📸 Signal received! Capturing...")
                        
                        # Clear buffer to ensure fresh frame
                        for _ in range(5):
                            cap.read()
                        
                        # Capture Actual Frame
                        ret, frame = cap.read()
                        if ret:
                            # Resize to 416x416 (Your Model Size)
                            frame_resized = cv2.resize(frame, (416, 416))
                            
                            # Save
                            filename = f"{SAVE_FOLDER}/img_{img_counter}.jpg"
                            cv2.imwrite(filename, frame_resized)
                            print(f"✅ Saved: {filename}")
                            img_counter += 1
                            
                            # Give a tiny delay to ensure save is done
                            time.sleep(0.1)
                            
                            # Tell Arduino to move to NEXT point
                            print("➡️ Sending 'n' (Next) to Arduino...")
                            ser.write(b'n')
                        else:
                            print("❌ Camera Frame Error")

                except Exception as e:
                    print(f"Read Error: {e}")

            # --- B. SHOW VIDEO FEED ---
            ret, frame = cap.read()
            if ret:
                # FIXED LINE IS HERE: Used cv2.FONT_HERSHEY_SIMPLEX instead of waitKey
                cv2.putText(frame, "Press 's' to Start Arduino", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Robot View", frame)
            
            # --- C. LISTEN FOR KEYBOARD INPUT ---
            # We call waitKey ONCE here and store it
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            
            elif key == ord('s'):
                print("\n🚀 's' pressed! Sending START signal to Arduino...")
                ser.write(b's')

    except KeyboardInterrupt:
        print("Stopping...")
    
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()