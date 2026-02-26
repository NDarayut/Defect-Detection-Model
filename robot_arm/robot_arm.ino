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

// Home position
#define HOME_BASE 60
#define HOME_SHOULDER 80
#define HOME_ELBOW 0
#define HOME_WRIST_ROT 60
#define HOME_WRIST_TILT 10

// Tilt limits - ADJUST THESE BASED ON YOUR ARM
#define TILT_MIN 0     // Camera looking forward/horizontal
#define TILT_MAX 45    // Camera looking steeply down (increased from 30)
// Typical range for your arm: 0-20° probably sufficient

// Current position
int b = HOME_BASE;
int s = HOME_SHOULDER;
int e = HOME_ELBOW;
int wr = HOME_WRIST_ROT;
int wt = HOME_WRIST_TILT;

void setup() {
  Serial.begin(9600);
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);
  delay(1000);
  
  Serial.println("Arm Control Ready");
  Serial.print("Tilt range: ");
  Serial.print(TILT_MIN);
  Serial.print("-");
  Serial.print(TILT_MAX);
  Serial.println(" (0=forward, 10=home/down)");
  Serial.println("Commands: b60 s80 e0 r60 t[0-45]");
  Serial.println("1=Test 2=Square h=Home 3=FindTilt");
  
  moveHome();
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    if (input.length() > 0) {
      char cmd = input.charAt(0);
      
      if (input.length() > 1) {
        String valStr = input.substring(1);
        int val = valStr.toInt();
        
        switch(cmd) {
          case 'b': 
            b = constrain(val, 0, 180);
            Serial.print("Base: ");
            Serial.println(b);
            moveServo(BASE_SERVO, b); 
            break;
          case 's': 
            s = constrain(val, 0, 180);
            Serial.print("Shoulder: ");
            Serial.println(s);
            moveServo(SHOULDER_SERVO, s); 
            break;
          case 'e': 
            e = constrain(val, 0, 180);
            Serial.print("Elbow: ");
            Serial.println(e);
            moveServo(ELBOW_SERVO, e); 
            break;
          case 'r': 
            wr = constrain(val, 0, 180);
            Serial.print("Wrist Rot: ");
            Serial.println(wr);
            moveServo(WRIST_ROT_SERVO, wr); 
            break;
          case 't': 
            wt = constrain(val, TILT_MIN, TILT_MAX);
            Serial.print("Wrist Tilt: ");
            Serial.print(wt);
            Serial.print("° (");
            if(wt == 0) Serial.print("forward");
            else if(wt < 10) Serial.print("slightly down");
            else if(wt == 10) Serial.print("home/down");
            else Serial.print("more down");
            Serial.println(")");
            moveServo(WRIST_TILT_SERVO, wt); 
            break;
          case '1': 
            testSequence(); 
            break;
          case '2': 
            drawSquare(); 
            break;
          case '3':
            findTiltLimits();
            break;
          case 'h': 
            Serial.println("Moving home...");
            moveHome(); 
            break;
          default:
            if(cmd >= '0' && cmd <= '9') {
              // Single digit commands already handled
            } else {
              Serial.println("Unknown command");
            }
        }
      } else {
        // Single character commands
        switch(cmd) {
          case '1': testSequence(); break;
          case '2': drawSquare(); break;
          case '3': findTiltLimits(); break;
          case 'h': 
            Serial.println("Moving home...");
            moveHome(); 
            break;
        }
      }
    }
  }
}

void moveServo(int servo, int angle) {
  angle = constrain(angle, 0, 180);
  int pulse = map(angle * 10, 0, 1800, SERVO_MIN, SERVO_MAX);
  pwm.setPWM(servo, 0, pulse);
  delay(50);
}

void moveHome() {
  b = HOME_BASE;
  s = HOME_SHOULDER;
  e = HOME_ELBOW;
  wr = HOME_WRIST_ROT;
  wt = HOME_WRIST_TILT;
  
  moveServo(BASE_SERVO, b);
  moveServo(SHOULDER_SERVO, s);
  moveServo(ELBOW_SERVO, e);
  moveServo(WRIST_ROT_SERVO, wr);
  moveServo(WRIST_TILT_SERVO, wt);
}

void testSequence() {
  Serial.println("Testing tilt range...");
  Serial.println("Finding what tilt angles work for your arm:");
  
  // Move to a good test position
  moveServo(BASE_SERVO, 60);
  moveServo(SHOULDER_SERVO, 85);
  moveServo(ELBOW_SERVO, 15);
  moveServo(WRIST_ROT_SERVO, 60);
  delay(1000);
  
  // Test different tilt angles
  Serial.println("\nTesting tilt angles (watch camera):");
  for(int tilt = 0; tilt <= 45; tilt += 5) {
    Serial.print("Tilt ");
    Serial.print(tilt);
    Serial.print("°: ");
    if(tilt == 0) Serial.println("Camera forward");
    else if(tilt == 10) Serial.println("Home position (should be down)");
    else if(tilt < 10) Serial.println("Slightly down");
    else Serial.println("More down");
    
    moveServo(WRIST_TILT_SERVO, tilt);
    delay(2000);
  }
  
  moveHome();
  Serial.println("Tilt test done!");
}

void findTiltLimits() {
  Serial.println("=== FINDING TILT LIMITS ===");
  Serial.println("Move arm manually, then test tilt angles");
  Serial.println("Find min/max tilt that keeps camera on surface");
  
  // Test at different arm heights
  int testPositions[3][3] = {
    {90, 30, "Top position"},      // Shoulder=90, Elbow=30
    {80, 0,  "Home position"},     // Shoulder=80, Elbow=0
    {70, 0,  "Low position"}       // Shoulder=70, Elbow=0
  };
  
  for(int pos = 0; pos < 3; pos++) {
    Serial.print("\n");
    Serial.println(testPositions[pos][2]);
    
    moveServo(BASE_SERVO, 60);
    moveServo(SHOULDER_SERVO, testPositions[pos][0]);
    moveServo(ELBOW_SERVO, testPositions[pos][1]);
    moveServo(WRIST_ROT_SERVO, 60);
    delay(1000);
    
    Serial.println("Test different tilts, find what keeps camera flat:");
    Serial.println("Press any key when ready for next position...");
    while(!Serial.available()) delay(100);
    Serial.read(); // Clear buffer
  }
  
  moveHome();
  Serial.println("\nTilt limit test complete!");
  Serial.println("Note: For square pattern:");
  Serial.println("- Top row (S=90,E=30): tilt ~0-5°");
  Serial.println("- Bottom row (S=80,E=0): tilt ~10°");
}

void drawSquare() {
  Serial.println("Drawing square with auto-tilt...");
  
  // Square corners with dynamic tilt calculation
  // Tilt decreases as arm goes up, increases as arm goes down
  int corners[5][5] = {
    {60, 80, 0, 60, 10},    // Home - tilt 10°
    {40, 80, 0, 50, 10},    // Bottom Right - same height, same tilt
    {40, 90, 30, 60, 5},    // Top Right - higher, less tilt (5°)
    {80, 90, 30, 70, 5},    // Top Left - higher, less tilt (5°)
    {80, 80, 0, 70, 10}     // Bottom Left - back to lower, more tilt (10°)
  };
  
  Serial.println("Tilt adjusts: 10° at bottom, 5° at top");
  
  for(int i = 0; i < 5; i++) {
    Serial.print("Point ");
    Serial.print(i);
    Serial.print(": Tilt=");
    Serial.print(corners[i][4]);
    Serial.println("°");
    
    b = corners[i][0];
    s = corners[i][1];
    e = corners[i][2];
    wr = corners[i][3];
    wt = corners[i][4];
    
    moveServo(BASE_SERVO, b);
    moveServo(SHOULDER_SERVO, s);
    moveServo(ELBOW_SERVO, e);
    moveServo(WRIST_ROT_SERVO, wr);
    moveServo(WRIST_TILT_SERVO, wt);
    
    delay(1000);
  }
  
  moveHome();
  Serial.println("Square done");
}