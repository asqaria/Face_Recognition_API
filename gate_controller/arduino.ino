#include <Servo.h>

Servo myservo;
int gate = 0;

void setup() {
  myservo.attach(7);
  Serial.begin(9600);
}

void loop() {
  myservo.write(0);
  if (Serial.available() > 0) {
    gate = Serial.read();
    if(gate == 48){
      myservo.write(60);
      delay(5000);
    }
  }
}
