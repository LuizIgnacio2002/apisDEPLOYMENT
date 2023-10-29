class ServoC {
  private:
    int angle;
    Servo servo;
    byte pin;
  public:
    ServoC(int pin) {
      this->pin = pin;
      angle = 0;
    }
    void begin() {
      servo.attach(pin);
      move(20);
    }
    void move(int angle) {
      this->angle = angle;
      this->servo.write(angle);
    }
};