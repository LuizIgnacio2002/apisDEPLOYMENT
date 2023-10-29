#include<ESP32Servo.h>
#include "UltraSonic.h"
#include "ServoC.h"
#include "Alarm.h"

UltraSonic ultrasonic(2, 4);
ServoC servo(23);
Alarm alarmC(5);

void setup() {
  // put your setup code here, to run once:
  ultrasonic.begin();
  servo.begin();
  alarmC.begin();
  Serial.begin(9600);
}
int pos = 20;
int f = 1;
int find = false;
int x = 0;
void loop() {
  // put your main code here, to run repeatedly:
  int d = ultrasonic.getDistance();
  find = d>=0 && d<=20;
  Serial.print("Distance: ");
  Serial.println(d);
  Serial.print("x: ");
  Serial.println(x);
  delay(500);
  if (find) {
    alarmC.tick();
    x = 0;
  } else {
    if (pos==20) {
      f = 1;
    }
    if (pos==140) {
      f = -1;
    }
    x++;
    if (x==5) {
      pos += f*40;
      servo.move(pos);
      x = 0;
    }    
  }
}
