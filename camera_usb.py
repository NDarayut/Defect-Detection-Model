import cv2
import os

def open_camera():
    # 1. Setup the save folder
    save_folder = "test_images"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Created folder: {save_folder}")

    # Initialize image counter for naming (img_0.jpg, img_1.jpg...)
    img_counter = 0

    # 2. Open Camera
    # You provided '1', keeping it as is. Change to '0' if it doesn't work.
    camera_index = 1 
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    print(f"Camera opened. Saving to '/{save_folder}'.")
    print("Controls:")
    print("  [SPACEBAR] - Take picture (saved as 416x416)")
    print("  [q]        - Quit")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break

        # Show the live feed (standard size)
        cv2.imshow('USB Camera Feed', frame)

        # Listen for key presses
        key = cv2.waitKey(1) & 0xFF

        # If 'q' is pressed, quit
        if key == ord('q'):
            break
        
        # If 'SPACEBAR' is pressed, save the image
        elif key == 32: # 32 is the ASCII code for Spacebar
            # Resize the captured frame to 416x416
            resized_frame = cv2.resize(frame, (416, 416))
            
            # Create the filename
            img_name = f"img_{img_counter}.jpg"
            path = os.path.join(save_folder, img_name)
            
            # Save the image
            cv2.imwrite(path, resized_frame)
            print(f"Saved: {path} (Resolution: 416x416)")
            
            # Increment counter so we don't overwrite the previous one
            img_counter += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    open_camera()