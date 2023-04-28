/*
Dan Cameron SEED Team 5
Lower Extremity Restraint
Housing Code for MKR WIFI 1010

Description:
This code controls a lower extremity restraint device, which is designed to restrain
the movement of a person's leg. It uses an accelerometer to detect excessive movement,
and a servo motor to lock the restraint in place. The device also records the acceleration
data to an SD card for later analysis.

Usage:
The device is activated by flipping a switch. When activated, the device will monitor the
acceleration data from the accelerometer. If the acceleration data exceeds certain
thresholds, the servo motor will be activated to lock the restraint in place. The device
will also record the acceleration data to an SD card. To deactivate the device, simply
flip the switch back to its original position.

Inputs:

Accelerometer data
Outputs:

Servo motor control signal
LED indicators
Acceleration data recorded to SD card
Revision history:
1.0 - Initial release

*/

#include <SD.h>  // Include the SD library for writing to an SD card


/*
unsigned int year = 2015;
byte month = 6;
byte day = 18;
byte hour = 7;
byte minute = 8;
byte second = 9;


// Helper function to convert current date and time to FAT format
void dateTime(uint16_t* date, uint16_t* time) {
  *date = FAT_DATE(year, month, day);
  *time = FAT_TIME(hour, minute, second);
}
*/

// Include necessary libraries
#include "Arduino.h"
#include "Arduino_BHY2Host.h"
#include <Servo.h>


// Define pin assignments
#define RED_LED 12
#define GREEN_LED 11
const int Switch = 1;
const int Servo_Pin = 13;
const int chipSelect = 4;


// Initialize necessary objects and variables
SensorXYZ accel(SENSOR_ID_ACC);
File sdcard_file;
char fileName[20];
Servo myservo;
int period = 10000;
int pos = 0;  
int lastCheck = 0;
unsigned long time_now = 0;
int lockdelay = 1000;
int16_t valueX;
int16_t valueY;
int16_t valueZ;


void setup() {
  // Set pin modes
  pinMode(RED_LED, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  pinMode(Switch, INPUT_PULLUP);

  // Start serial communication
  Serial.begin(115200);

  // Initialize servo
  myservo.attach(Servo_Pin);

  // Initialize SD card
  Serial.print("Initializing SD card...");
  if (!SD.begin(chipSelect)) {  // If SD card cannot be initialized
    Serial.println("SD.begin failed");
    while (1);  // Loop indefinitely
  }
  Serial.println("card initialized.");

  // Initialize sensor
  BHY2Host.begin(false, NICLA_VIA_BLE);
  accel.begin();
}


void loop() {
  // Get current time
  time_now = millis();

  // Check for new data from sensor
  BHY2Host.update();
  valueX = accel.x();
  valueY = accel.y();
  valueZ = accel.z();

  // Check sensor values every 100ms
  if (millis() - lastCheck >= 100) {
    lastCheck = millis();

    // Print sensor values to serial monitor
    Serial.print(valueX);
    Serial.print(",");
    Serial.print(valueY);
    Serial.print(",");
    Serial.println(valueZ);

    // Write sensor values to SD card
    sdcard_file = SD.open("standby.csv", FILE_WRITE);
    while (sdcard_file) {
      sdcard_file.print(valueX);
      sdcard_file.print(",");
      sdcard_file.print(valueY);
      sdcard_file.print(",");
      sdcard_file.println(valueZ);
      sdcard_file.close();
    }

    // Check switch state and control servo accordingly
    int switchState = digitalRead(Switch);
    if (switchState == LOW) {  // Switch is pressed
      digitalWrite(GREEN_LED, LOW);
digitalWrite(RED_LED, LOW);
myservo.write(180); // Move servo to open position
}

if (switchState == HIGH) {  // Switch is not pressed
  digitalWrite(GREEN_LED, HIGH);
  digitalWrite(RED_LED, LOW);
  if (abs(valueX) >= abs(3000) || abs(valueY) >= abs(3000) || abs(valueZ) >= abs(6000)) {  // If acceleration exceeds threshold
    myservo.write(0);  // Move servo to closed position
    digitalWrite(RED_LED, HIGH);
    digitalWrite(GREEN_LED, LOW);
    while (millis() < time_now + lockdelay) {  // Delay to lock restraint in place for specified time
    }
  } else {  // Acceleration is within acceptable range
    digitalWrite(RED_LED, LOW);
    digitalWrite(GREEN_LED, HIGH);
    myservo.write(180);  // Move servo to open position
    while (millis() < time_now + lockdelay) {  // Delay to keep restraint open for specified time
    }
  }
}

}
}

void ServoOpen() {
myservo.write(180);
}

void ServoClose() {
myservo.write(0);
}

/*

Function: dateTime
Sets the date and time for the SD card file creation.
*date: pointer to the date value to be set
*time: pointer to the time value to be set
returns: void
/
void dateTime(uint16_t date, uint16_t* time) {
*date = FAT_DATE(year, month, day);
*time = FAT_TIME(hour, minute, second);
}
*/
