/*
 * AgitationMonitorAccelerometer.ino - Lower Extremity Restraint 
 * Author: Dan Cameron
 * Date: 4/28/2023
 *
Starts the bluetooth connection for the nicla sense ME, enabling the 
Arduino MKR to connect directly to the nicla sense. BHY2.begin() starts
the connetion to the host board, which is the Arduino MKR.
 */


#include "Arduino.h"
#include "Arduino_BHY2.h"

// Set DEBUG to true in order to enable debug print
#define DEBUG false

void setup()
{
#if DEBUG
  Serial.begin(115200); 
  BHY2.debug(Serial);;
#endif

  BHY2.begin();
}

void loop()
{
  // Update and then sleep
  BHY2.update(100);
}
