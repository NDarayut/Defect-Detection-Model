#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVO_FREQ 50
#define SERVO_MIN 150
#define SERVO_MAX 600

// Servo channels
#define BASE_SERVO 0
#define SHOULDER_SERVO 1
#define ELBOW_SERVO 2
#define WRIST_ROT_SERVO 3
#define WRIST_TILT_SERVO 4

// Your square corners
struct Corner {
  int base;
  int shoulder;
  int elbow;
  int wrist_rot;
  int tilt;
  String name;
};

// Define corners in scanning order: BL, BR, TR, TL (counter-clockwise)
Corner corners[4] = {
  {80, 80, 0, 70, 10, "BOTTOM LEFT"},    // Start here
  {40, 80, 0, 50, 10, "BOTTOM RIGHT"},
  {40, 90, 30, 60, 40, "TOP RIGHT"},
  {80, 90, 30, 70, 40, "TOP LEFT"}
};

// Scanning parameters - CHANGED TO VARIABLES
int SCAN_ROWS = 5;           // Number of rows in scan
int SCAN_COLS = 5;           // Number of columns in scan
int SCAN_DELAY = 800;        // Time at each scan point (ms) - adjustable
int MOVE_SPEED = 40;         // Movement speed between points (ms delay per step)

// Current position
int current_base = 60;
int current_shoulder = 80;
int current_elbow = 0;
int current_wrist_rot = 60;
int current_tilt = 10;

void setup() {
  Serial.begin(9600);
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);
  delay(1000);
  
  Serial.println("=== ROW-BY-ROW SCANNING ===");
  Serial.print("Grid: ");
  Serial.print(SCAN_ROWS);
  Serial.print(" rows x ");
  Serial.print(SCAN_COLS);
  Serial.println(" columns");
  Serial.print("Scan delay: ");
  Serial.print(SCAN_DELAY);
  Serial.println(" ms per point");
  Serial.print("Move speed: ");
  Serial.print(MOVE_SPEED);
  Serial.println(" ms per step");
  Serial.println("\nCommands:");
  Serial.println("s : Start scanning");
  Serial.println("p : Pause/Resume");
  Serial.println("h : Home position");
  Serial.println("1-9 : Set scan delay (1=fast, 9=slow)");
  Serial.println("a : Set grid size (e.g., a4x4)");
  Serial.println("========================");
  
  moveHome();
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    
    switch(cmd) {
      case 's': startScanning(); break;
      case 'p': Serial.println("Pause - restart with 's'"); break;
      case 'h': moveHome(); break;
      case '1': SCAN_DELAY = 300; Serial.println("Fast scan: 300ms delay"); break;
      case '2': SCAN_DELAY = 400; Serial.println("Medium-fast scan: 400ms delay"); break;
      case '3': SCAN_DELAY = 500; Serial.println("Medium scan: 500ms delay"); break;
      case '4': SCAN_DELAY = 600; Serial.println("Medium-slow scan: 600ms delay"); break;
      case '5': SCAN_DELAY = 700; Serial.println("Default scan: 700ms delay"); break;
      case '6': SCAN_DELAY = 800; Serial.println("Slow scan: 800ms delay"); break;
      case '7': SCAN_DELAY = 1000; Serial.println("Very slow scan: 1000ms delay"); break;
      case '8': SCAN_DELAY = 1500; Serial.println("Debug scan: 1500ms delay"); break;
      case '9': SCAN_DELAY = 2000; Serial.println("Ultra slow scan: 2000ms delay"); break;
      case 'a': setGridSize(); break;
    }
  }
}

void setGridSize() {
  Serial.println("Enter grid size (e.g., 4x4 or 6x6):");
  
  // Wait for input
  delay(100);
  while(Serial.available() == 0) {
    delay(10);
  }
  
  String input = Serial.readStringUntil('\n');
  input.trim();
  
  // Parse input like "4x4" or "6x6"
  int xIndex = input.indexOf('x');
  if(xIndex > 0) {
    String rowsStr = input.substring(0, xIndex);
    String colsStr = input.substring(xIndex + 1);
    
    SCAN_ROWS = rowsStr.toInt();
    SCAN_COLS = colsStr.toInt();
    
    // Validate
    if(SCAN_ROWS < 2) SCAN_ROWS = 2;
    if(SCAN_COLS < 2) SCAN_COLS = 2;
    if(SCAN_ROWS > 10) SCAN_ROWS = 10;
    if(SCAN_COLS > 10) SCAN_COLS = 10;
    
    Serial.print("Grid set to: ");
    Serial.print(SCAN_ROWS);
    Serial.print("x");
    Serial.println(SCAN_COLS);
  }
}

void moveServo(int servo, int angle) {
  angle = constrain(angle, 0, 180);
  int pulse = map(angle * 10, 0, 1800, SERVO_MIN, SERVO_MAX);
  pwm.setPWM(servo, 0, pulse);
}

void moveToPosition(int base, int shoulder, int elbow, int wrist_rot, int tilt) {
  // Calculate number of steps needed based on largest movement
  int base_steps = abs(base - current_base);
  int shoulder_steps = abs(shoulder - current_shoulder);
  int elbow_steps = abs(elbow - current_elbow);
  int wrist_steps = abs(wrist_rot - current_wrist_rot);
  int tilt_steps = abs(tilt - current_tilt);
  
  int max_steps = max(base_steps, max(shoulder_steps, max(elbow_steps, max(wrist_steps, tilt_steps))));
  max_steps = max(max_steps, 1); // Minimum 1 step
  
  // Smooth movement with interpolation
  for (int i = 0; i <= max_steps; i++) {
    float t = (float)i / max_steps;
    
    int b = current_base + (base - current_base) * t;
    int s = current_shoulder + (shoulder - current_shoulder) * t;
    int e = current_elbow + (elbow - current_elbow) * t;
    int w = current_wrist_rot + (wrist_rot - current_wrist_rot) * t;
    int ti = current_tilt + (tilt - current_tilt) * t;
    
    moveServo(BASE_SERVO, b);
    moveServo(SHOULDER_SERVO, s);
    moveServo(ELBOW_SERVO, e);
    moveServo(WRIST_ROT_SERVO, w);
    moveServo(WRIST_TILT_SERVO, ti);
    
    delay(MOVE_SPEED); // Control movement speed
  }
  
  // Update current position
  current_base = base;
  current_shoulder = shoulder;
  current_elbow = elbow;
  current_wrist_rot = wrist_rot;
  current_tilt = tilt;
}

void moveHome() {
  Serial.println("Moving to home position...");
  moveToPosition(60, 80, 0, 60, 10);
}

// Calculate position for a specific grid point
void calculateGridPoint(int row, int col, int &base, int &shoulder, int &elbow, int &wrist_rot, int &tilt) {
  float row_frac = (float)row / (SCAN_ROWS - 1);
  float col_frac = (float)col / (SCAN_COLS - 1);
  
  // Get the four corners
  int bl_base = corners[0].base;
  int bl_shoulder = corners[0].shoulder;
  int bl_elbow = corners[0].elbow;
  int bl_wrist_rot = corners[0].wrist_rot;
  int bl_tilt = corners[0].tilt;
  
  int br_base = corners[1].base;
  int br_shoulder = corners[1].shoulder;
  int br_elbow = corners[1].elbow;
  int br_wrist_rot = corners[1].wrist_rot;
  int br_tilt = corners[1].tilt;
  
  int tr_base = corners[2].base;
  int tr_shoulder = corners[2].shoulder;
  int tr_elbow = corners[2].elbow;
  int tr_wrist_rot = corners[2].wrist_rot;
  int tr_tilt = corners[2].tilt;
  
  int tl_base = corners[3].base;
  int tl_shoulder = corners[3].shoulder;
  int tl_elbow = corners[3].elbow;
  int tl_wrist_rot = corners[3].wrist_rot;
  int tl_tilt = corners[3].tilt;
  
  // Interpolate horizontally at bottom
  float bottom_base = bl_base + (br_base - bl_base) * col_frac;
  float bottom_shoulder = bl_shoulder + (br_shoulder - bl_shoulder) * col_frac;
  float bottom_elbow = bl_elbow + (br_elbow - bl_elbow) * col_frac;
  float bottom_wrist_rot = bl_wrist_rot + (br_wrist_rot - bl_wrist_rot) * col_frac;
  float bottom_tilt = bl_tilt + (br_tilt - bl_tilt) * col_frac;
  
  // Interpolate horizontally at top
  float top_base = tl_base + (tr_base - tl_base) * col_frac;
  float top_shoulder = tl_shoulder + (tr_shoulder - tl_shoulder) * col_frac;
  float top_elbow = tl_elbow + (tr_elbow - tl_elbow) * col_frac;
  float top_wrist_rot = tl_wrist_rot + (tr_wrist_rot - tl_wrist_rot) * col_frac;
  float top_tilt = tl_tilt + (tr_tilt - tl_tilt) * col_frac;
  
  // Interpolate vertically
  base = top_base + (bottom_base - top_base) * row_frac;
  shoulder = top_shoulder + (bottom_shoulder - top_shoulder) * row_frac;
  elbow = top_elbow + (bottom_elbow - top_elbow) * row_frac;
  wrist_rot = top_wrist_rot + (bottom_wrist_rot - top_wrist_rot) * row_frac;
  tilt = top_tilt + (bottom_tilt - top_tilt) * row_frac;
}

void startScanning() {
  Serial.println("=== STARTING ROW-BY-ROW SCAN ===");
  Serial.print("Grid size: ");
  Serial.print(SCAN_ROWS);
  Serial.print("x");
  Serial.println(SCAN_COLS);
  Serial.print("Scan delay: ");
  Serial.print(SCAN_DELAY);
  Serial.println(" ms per point");
  
  Serial.println("Moving to start position...");
  
  // Move to bottom-left corner first
  moveToPosition(corners[0].base, corners[0].shoulder, corners[0].elbow, 
                 corners[0].wrist_rot, corners[0].tilt);
  delay(1000);
  
  int total_points = SCAN_ROWS * SCAN_COLS;
  int point_count = 0;
  
  // Row-by-row scanning with serpentine pattern (no overlap)
  for(int row = 0; row < SCAN_ROWS; row++) {
    Serial.print("\n=== ROW ");
    Serial.print(row + 1);
    Serial.print("/");
    Serial.print(SCAN_ROWS);
    Serial.println(" ===");
    
    // Determine scanning direction (serpentine pattern)
    bool left_to_right = (row % 2 == 0);
    
    if(left_to_right) {
      // Scan left to right
      for(int col = 0; col < SCAN_COLS; col++) {
        scanPoint(row, col, ++point_count, total_points);
      }
    } else {
      // Scan right to left
      for(int col = SCAN_COLS - 1; col >= 0; col--) {
        scanPoint(row, col, ++point_count, total_points);
      }
    }
  }
  
  Serial.println("\n=== SCAN COMPLETE ===");
  Serial.print("Scanned ");
  Serial.print(total_points);
  Serial.println(" points without overlap");
  
  moveHome();
}

void scanPoint(int row, int col, int point_num, int total_points) {
  int base, shoulder, elbow, wrist_rot, tilt;
  calculateGridPoint(row, col, base, shoulder, elbow, wrist_rot, tilt);
  
  Serial.print("Point ");
  Serial.print(point_num);
  Serial.print("/");
  Serial.print(total_points);
  Serial.print(" [R");
  Serial.print(row);
  Serial.print("C");
  Serial.print(col);
  Serial.print("]: ");
  Serial.print("B=");
  Serial.print(base);
  Serial.print(" S=");
  Serial.print(shoulder);
  Serial.print(" E=");
  Serial.print(elbow);
  Serial.print(" T=");
  Serial.print(tilt);
  Serial.println("°");
  
  // Move to the calculated position
  moveToPosition(base, shoulder, elbow, wrist_rot, tilt);
  
  // Wait at scan position (for camera capture/processing)
  delay(SCAN_DELAY);
  
  // Optional: Add camera trigger code here
  // triggerCamera();
}