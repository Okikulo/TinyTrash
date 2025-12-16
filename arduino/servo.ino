#include <Adafruit_PWMServoDriver.h>
#include <Wire.h>

Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);

#define SERVOMIN 110 // pulse for 0°
#define SERVOMAX 510 // pulse for 180°

int servos[4] = {2, 1, 0, 3}; // PCA9685 channels for each servo

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("Sequential 4 Servo Control");

  pca.begin();
  pca.setPWMFreq(50);
  delay(10);
}

void setServoAngle(int channel, int angle) {
  int pulse = map(angle, 0, 180, SERVOMIN, SERVOMAX);
  pca.setPWM(channel, 0, pulse);
}

void moveServo_0_to_90(int channel) {
  Serial.print("Servo ");
  Serial.print(channel);
  Serial.println(" → Opening (0° → 90°)");

  for (int angle = 0; angle <= 90; angle += 5) {
    setServoAngle(channel, angle);
    delay(10);
  }
}

void moveServo_90_to_0(int channel) {
  Serial.print("Servo ");
  Serial.print(channel);
  Serial.println(" → Closing (90° → 0°)");

  for (int angle = 90; angle >= 0; angle -= 5) {
    setServoAngle(channel, angle);
    delay(10);
  }
}

void loop() {
  // Move servos one by one
  for (int i = 0; i < 4; i++) {
    int ch = servos[i];

    moveServo_0_to_90(ch); // open
    delay(500);

    moveServo_90_to_0(ch); // close
    delay(1000);           // wait before next servo
  }
}
