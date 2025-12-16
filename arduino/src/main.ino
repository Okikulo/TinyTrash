#include <Adafruit_PWMServoDriver.h>
#include <TFT_eSPI.h>
#include <Wire.h>

// LCD
TFT_eSPI tft;

// Servo controller
Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);

// Servo settings
#define SERVOMIN 110 // 0°
#define SERVOMAX 510 // 180°

// Servo channel mapping (alphabetical order)
#define SERVO_GLASS 2
#define SERVO_METAL 1
#define SERVO_PAPER 0
#define SERVO_PLASTIC 3

// Colors for categories
#define COLOR_GLASS TFT_CYAN
#define COLOR_METAL 0xC618 // Silver/Gray
#define COLOR_PAPER TFT_YELLOW
#define COLOR_PLASTIC TFT_BLUE

void setup() {
  Serial.begin(115200);

  // Initialize LCD
  tft.init();
  tft.setRotation(3);
  tft.fillScreen(TFT_BLACK);
  tft.setTextColor(TFT_WHITE);
  tft.setTextSize(2);
  tft.setCursor(20, 100);
  tft.println("TinyTrash Ready");
  tft.setCursor(20, 130);
  tft.println("Waiting for data...");

  // Initialize servo controller
  pca.begin();
  pca.setPWMFreq(50);
  delay(10);

  // Close all servos initially
  for (int i = 0; i < 5; i++) {
    setServoAngle(i, 0);
  }

  Serial.println("TinyTrash - LCD + Servo Ready");
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    data.trim();

    // Parse "CATEGORY:CONFIDENCE"
    int colonIndex = data.indexOf(':');
    if (colonIndex > 0) {
      String category = data.substring(0, colonIndex);
      String confidence = data.substring(colonIndex + 1);

      category.toUpperCase();

      // Get color and servo channel
      uint16_t color = TFT_WHITE;
      int servoChannel = -1;

      if (category == "GLASS") {
        color = COLOR_GLASS;
        servoChannel = SERVO_GLASS;
      } else if (category == "METAL") {
        color = COLOR_METAL;
        servoChannel = SERVO_METAL;
      } else if (category == "PAPER") {
        color = COLOR_PAPER;
        servoChannel = SERVO_PAPER;
      } else if (category == "PLASTIC") {
        color = COLOR_PLASTIC;
        servoChannel = SERVO_PLASTIC;
      }

      // Display on LCD
      displayPrediction(category, confidence, color);

      // Open corresponding bin
      if (servoChannel >= 0) {
        openBin(servoChannel, category);
      }

      Serial.print("Received: ");
      Serial.print(category);
      Serial.print(" - ");
      Serial.print(confidence);
      Serial.println("%");
    }
  }
}

void displayPrediction(String category, String confidence, uint16_t color) {
  tft.fillScreen(TFT_BLACK);

  // Title
  tft.setTextColor(TFT_WHITE);
  tft.setTextSize(2);
  tft.setCursor(60, 20);
  tft.println("CLASSIFICATION");

  // Category (large, colored)
  tft.setTextColor(color);
  tft.setTextSize(4);
  int xPos = (320 - category.length() * 24) / 2; // Center
  tft.setCursor(xPos, 80);
  tft.println(category);

  // Confidence
  tft.setTextColor(TFT_WHITE);
  tft.setTextSize(3);
  String confText = confidence + "%";
  xPos = (320 - confText.length() * 18) / 2; // Center
  tft.setCursor(xPos, 140);
  tft.println(confText);

  // Confidence bar
  float conf = confidence.toFloat();
  int barWidth = (int)(280 * conf / 100.0);
  tft.drawRect(20, 190, 280, 30, TFT_WHITE);
  tft.fillRect(20, 190, barWidth, 30, color);
}

void setServoAngle(int channel, int angle) {
  int pulse = map(angle, 0, 180, SERVOMIN, SERVOMAX);
  pca.setPWM(channel, 0, pulse);
}

void openBin(int channel, String category) {
  Serial.print("Opening ");
  Serial.print(category);
  Serial.println(" bin...");

  // Open (0° → 90°)
  for (int angle = 0; angle <= 90; angle += 5) {
    setServoAngle(channel, angle);
    delay(15);
  }

  // Hold open for 3 seconds
  delay(5000);

  // Close (90° → 0°)
  Serial.print("Closing ");
  Serial.print(category);
  Serial.println(" bin...");

  for (int angle = 90; angle >= 0; angle -= 5) {
    setServoAngle(channel, angle);
    delay(15);
  }

  Serial.println("Ready for next item");
}
