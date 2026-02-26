import os

# --- CONFIGURATION ---
# Change these paths to your actual folders
IMAGES_DIR = "golden_dataset/train/images"
LABELS_DIR = "golden_dataset/train/labels"

def create_missing_labels():
    # 1. Get list of all images
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Found {len(image_files)} images.")
    
    created_count = 0
    
    for img_file in image_files:
        # Construct expected label filename (e.g., image.jpg -> image.txt)
        label_filename = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(LABELS_DIR, label_filename)
        
        # 2. Check if the text file exists
        if not os.path.exists(label_path):
            # 3. If not, create an empty one
            with open(label_path, 'w') as f:
                pass # Do nothing, just create empty file
            print(f"Created empty label: {label_filename}")
            created_count += 1
            
    print(f"\nDone! Created {created_count} empty label files.")
    print("Your dataset is now ready for 'Negative Sample' training.")

if __name__ == "__main__":
    # Create the labels folder if it doesn't exist
    if not os.path.exists(LABELS_DIR):
        os.makedirs(LABELS_DIR)
        
    create_missing_labels()