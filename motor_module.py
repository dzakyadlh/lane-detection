# Motor driver: L298N

import RPi.GPIO as GPIO
from time import sleep
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

class Motor():
    def __init__(self, EnaA, In1A, In2A, EnaB, In1B, In2B):
        self.EnaA = EnaA
        self.In1A = In1A
        self.In2A = In2A
        self.EnaB = EnaB
        self.In1B = In1B
        self.In2B = In2B
        GPIO.setup(self.EnaA, GPIO.OUT)
        GPIO.setup(self.In1A, GPIO.OUT)
        GPIO.setup(self.In2A, GPIO.OUT)
        GPIO.setup(self.EnaB, GPIO.OUT)
        GPIO.setup(self.In1B, GPIO.OUT)
        GPIO.setup(self.In2B, GPIO.OUT)
        self.pwm_A = GPIO.PWM(self.EnaA, 100)
        self.pwm_A.start(0)
        self.pwm_B = GPIO.PWM(self.EnaB, 100)
        self.pwm_B.start(0)

    def move(self, speed=0.5, turn=0, t=0):
        speed *= 100
        turn *= 100
        left_speed = speed - turn
        right_speed = speed + turn
        if left_speed>100: left_speed=100
        elif left_speed<-100: left_speed=-100
        if right_speed>100:right_speed=100
        elif right_speed<-100:right_speed=-100

        self.pwm_A.ChangeDutyCycle(abs(left_speed))
        self.pwm_B.ChangeDutyCycle(abs(right_speed))

        if left_speed>0:
            GPIO.output(self.In1A, GPIO.HIGH)
            GPIO.output(self.In2A, GPIO.LOW)
        else:
            GPIO.output(self.In1A, GPIO.LOW)
            GPIO.output(self.In2A, GPIO.HIGH)

        if right_speed>0:
            GPIO.output(self.In1A, GPIO.HIGH)
            GPIO.output(self.In2A, GPIO.LOW)
        else:
            GPIO.output(self.In1A, GPIO.LOW)
            GPIO.output(self.In2A, GPIO.HIGH)
        
        sleep(t)
    
    def stop(self, t=0):
        self.pwm_A.ChangeDutyCycle(0)
        self.pwm_B.ChangeDutyCycle(0)
        sleep(t)

def main():
    motor.move(0.6, 0, 2)
    motor.stop(2)
    motor.move(-0.5, 0.2, 2)
    motor.stop(2)

if __name__ == "__main__":
    motor = Motor(2, 3, 4, 17, 22, 27)
    main()