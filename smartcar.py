import cv2
import RPi.GPIO as GPIO
from time import sleep, time
from ultralytics import YOLO

# Config variables
speed = 10

# Setup GPIO settings
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Give ports to every motor
(Ena1, In11, In12) = (2, 3, 4)
(Ena2, In21, In22) = (17, 27, 22)
(Ena3, In31, In32) = (10, 9, 11)
(Ena4, In41, In42) = (14, 15, 18)

# Create a list for ports from the same category
enables = [Ena1, Ena2, Ena3, Ena4]
lefts = [In11, In21, In31, In41]
rights = [In12, In22, In32, In42]
pwms = []

# Setup all ports as out
for (enable, left, right) in zip(enables, lefts, rights):
    GPIO.setup(enable, GPIO.OUT)
    GPIO.setup(left, GPIO.OUT)
    GPIO.setup(right, GPIO.OUT)

# Setup the enables
for enable in enables:
    pwm = GPIO.PWM(enable, speed)
    pwm.start(0)
    pwms.append(pwm)

going_forward = False

# Smooth transition going and stopping
def set_motor_speed(speed):
    for pwm in pwms:
        pwm.ChangeDutyCycle(speed)

def gradient_descent(step):
    global speed

    for i in range(7):
        set_motor_speed(speed)
        speed += step
        sleep(0.05)

# Setup electricity flow
def set_direction():
    for left in lefts:
        GPIO.output(left, GPIO.HIGH)

    for right in rights:
        GPIO.output(right, GPIO.LOW)

def forward():
    set_direction()

    global going_forward

    if not going_forward:
        gradient_descent(10)

    going_forward = 1

def stop():
    global going_forward

    if going_forward:
        gradient_descent(-10)

    going_forward = 0

    set_motor_speed(0)

# Stop before running the program
stop()

# Loading model and defining labels
model = YOLO('weights/best.pt')
labels = {
    0: 'Green Light',
    1: 'Red Light',
    2: 'Speed Limit 10',
    3: 'Speed Limit 100',
    4: 'Speed Limit 110',
    5: 'Speed Limit 120',
    6: 'Speed Limit 20',
    7: 'Speed Limit 30',
    8: 'Speed Limit 40',
    9: 'Speed Limit 50',
    10: 'Speed Limit 60',
    11: 'Speed Limit 70',
    12: 'Speed Limit 80',
    13: 'Speed Limit 90',
    14: 'Stop',
    }

# Getting access to the camera
vid = cv2.VideoCapture(0)
print('Camera ready')

# Doing a dummy read and infer
(ret, frame) = vid.read()
flipped = cv2.flip(cv2.flip(cv2.resize(frame, dsize=(416, 416)), 0), 1)
results = model.predict(flipped, verbose=False, stream=True)
is_at_red_light = False

while True:
    if not is_at_red_light:
        forward()

    # Read image and infer
    (ret, frame) = vid.read()
    flipped = cv2.flip(cv2.flip(cv2.resize(frame, dsize=(416, 416)), 0), 1)
    results = model.predict(flipped, half=False, verbose=False)
    
    # Process results
    for result in results:
        box = result.boxes
        if len(box.cls) != 0 and len(box.xyxy.tolist()[0]) == 4 and float(box.conf[0]) > 0.75:
            label = labels[int(box.cls[0])]

            x1, y1, x2, y2 = [int(res) for res in
                                box.xyxy.tolist()[0]]

            # Result printing
            print('Found ', label, ' at ', x1, y1, x2, y2, ' and confidence ', float(box.conf[0]))
            print('Procent area covered ', ((x2 - x1) * (y2 - y1)) / (416 * 416), '\n')

            # Sign based behaviour
            if label == 'Stop':
                while True:
                    stop()
            elif label == 'Red Light':
                is_at_red_light = True
                stop()
            elif label == 'Green Light':
                is_at_red_light = False
            elif label == 'Speed Limit 70' and not is_at_red_light:
                set_motor_speed(100)
            elif label == 'Speed Limit 50' and not is_at_red_light:
                set_motor_speed(80) 
