/* Edge Impulse ingestion SDK
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

//This code has been modified from an example library code - after library creation in Edge Impulse
//Inference over Bluetooth was also modified from https://github.com/edgeimpulse/example-SparkFun-MicroMod-nRF52840#going-further-send-your-inference-over-bluetooth


// Import TensorFlow stuff
//#include <TensorFlowLite.h>

// we are using a variety of functions from the TensorFlowLite library and need to reference them here
//#include "tensorflow/lite/micro/all_ops_resolver.h"
//#include "tensorflow/lite/c/common.h"
//#include "tensorflow/lite/micro/micro_error_reporter.h"
//#include "tensorflow/lite/micro/micro_interpreter.h"
//#include "tensorflow/lite/schema/schema_generated.h"
//#include "tensorflow/lite/version.h"


// Define the variables used by TFLite (pointers)
//tflite::ErrorReporter* error_reporter = nullptr;
//const tflite::Model* model = nullptr;
//tflite::MicroInterpreter* interpreter = nullptr;
//TfLiteTensor* input = nullptr;
//TfLiteTensor* output = nullptr;

//#include <wm90-project-1_inferencing.h>
//#include <fall2_inferencing.h>
//#include <fall3_inferencing.h>
//#include <fall3_rnn_inferencing.h>
//#include <rnn_model.h>
#include <fall3_filtered_inferencing.h>
#include <Arduino_LSM9DS1.h> //Click here to get the library: https://www.arduino.cc/reference/en/libraries/arduino_lsm9ds1/
#include <ArduinoBLE.h>




BLEService inferenceService("19B10010-E8F2-537E-4F6C-D104768A1214"); // create service
BLEStringCharacteristic inferenceCharacteristic("19B10013-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify,56);



BLEByteCharacteristic ledCharacteristic("19B10011-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite);
// create button characteristic and allow remote device to get notifications
BLEByteCharacteristic buttonCharacteristic("19B10012-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify);


/* Constant defines -------------------------------------------------------- */
#define CONVERT_G_TO_MS2    9.80665f
#define MAX_ACCEPTED_RANGE  2.0f        // starting 03/2022, models are generated setting range to +-2, but this example use Arudino library which set range to +-4g. If you are using an older model, ignore this value and use 4.0f instead

/*
 ** NOTE: If you run into TFLite arena allocation issue.
 **
 ** This may be due to may dynamic memory fragmentation.
 ** Try defining "-DEI_CLASSIFIER_ALLOCATION_STATIC" in boards.local.txt (create
 ** if it doesn't exist) and copy this file to
 ** `<ARDUINO_CORE_INSTALL_PATH>/arduino/hardware/<mbed_core>/<core_version>/`.
 **
 ** See
 ** (https://support.arduino.cc/hc/en-us/articles/360012076960-Where-are-the-installed-cores-located-)
 ** to find where Arduino installs cores on your machine.
 **
 ** If the problem persists then there's not enough memory for this model and application.
 */

/* Private variables ------------------------------------------------------- */
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal

/**
* @brief      Arduino setup function
*/
void setup()
{
  
    // put your setup code here, to run once:
    Serial.begin(115200);
    // comment out the below line to cancel the wait for USB connection (needed for native USB)
    //while (!Serial);
    Serial.println("Edge Impulse Inferencing Demo");

    if (!BLE.begin()) {   // initialize BLE
      Serial.println("starting BLE failed!");
      while (1);
    }

    BLE.setLocalName("Test");  // Set device name 
    //Set the minimum and maximum desired connection intervals in units of 1.25 ms.
    //Bluetooth LE desired connection Interval 160ms - 200ms. Central has the final word! 
    //BLE.setConnectionInterval(160, 200);
    BLE.setAdvertisedService(inferenceService); // Advertise service
    inferenceService.addCharacteristic(inferenceCharacteristic); // Add characteristic to service
    
    inferenceService.addCharacteristic(ledCharacteristic);
    inferenceService.addCharacteristic(buttonCharacteristic);
    
    BLE.addService(inferenceService); // Add service
    inferenceCharacteristic.writeValue("inference"); // Set first value string
    //Set the advertising interval in units of 0.625 ms
    //Bluetooth LE advertising interval around 50ms
    //BLE.setAdvertisingInterval(80);
    BLE.advertise();  // Start advertising

    ledCharacteristic.writeValue(0);
    buttonCharacteristic.writeValue(0);



    if (!IMU.begin()) {
        ei_printf("Failed to initialize IMU!\r\n");
    }
    else {
        ei_printf("IMU initialized\r\n");
    }

    if (EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME != 3) {
        ei_printf("ERR: EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME should be equal to 3 (the 3 sensor axes)\n");
        return;
    }

    
}

/**
 * @brief Return the sign of the number
 * 
 * @param number 
 * @return int 1 if positive (or 0) -1 if negative
 */
float ei_get_sign(float number) {
    return (number >= 0.0) ? 1.0 : -1.0;
}

/**
* @brief      Get data and run inferencing
*
* @param[in]  debug  Get debug info if true
*/
void loop()
{

    BLE.poll();

    // wait for a BLE central
    BLEDevice central = BLE.central();
    String inferenceResult = "";

    // if a central is connected to the peripheral:
    if (central.discoverAttributes()) {
      // turn on the LED to indicate the connection:
      //digitalWrite(LED_BUILTIN, HIGH);
      ei_printf("\nStarting inferencing in 2 seconds...\n");

    }

    ei_printf("\nStarting inferencing in 2 seconds...\n");

    //delay(2000);

    ei_printf("Sampling...\n");

    // Allocate a buffer here for the values we'll read from the IMU
    float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = { 0 };

    for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += 3) {
        // Determine the next tick (and then sleep later)
        uint64_t next_tick = micros() + (EI_CLASSIFIER_INTERVAL_MS * 1000);

        IMU.readAcceleration(buffer[ix], buffer[ix + 1], buffer[ix + 2]);

        for (int i = 0; i < 3; i++) {
            if (fabs(buffer[ix + i]) > MAX_ACCEPTED_RANGE) {
                buffer[ix + i] = ei_get_sign(buffer[ix + i]) * MAX_ACCEPTED_RANGE;
            }
        }

        buffer[ix + 0] *= CONVERT_G_TO_MS2;
        buffer[ix + 1] *= CONVERT_G_TO_MS2;
        buffer[ix + 2] *= CONVERT_G_TO_MS2;

        delayMicroseconds(next_tick - micros());
    }

    // Turn the raw buffer in a signal which we can the classify
    signal_t signal;
    int err = numpy::signal_from_buffer(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
    if (err != 0) {
        ei_printf("Failed to create signal from buffer (%d)\n", err);
        return;
    }

    // Run the classifier
    ei_impulse_result_t result = { 0 };

    err = run_classifier(&signal, &result, debug_nn);
    if (err != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", err);
        return;
    }

    // print the predictions
    ei_printf("Predictions ");
    ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
        result.timing.dsp, result.timing.classification, result.timing.anomaly);
    ei_printf(": \n");
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        ei_printf("    %s: %.5f\n", result.classification[ix].label, result.classification[ix].value);
        if(result.classification[ix].value >0.0 && inferenceResult != result.classification[ix].label){
            inferenceResult = result.classification[ix].label;
            sendInferenceOverBLE(inferenceResult);
          } 
    }
    
#if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("    anomaly score: %.3f\n", result.anomaly);
#endif
}


void sendInferenceOverBLE(String inferenceResult) {

  inferenceCharacteristic.writeValue(inferenceResult); // Write value
}

  /*
  
#if (DEBUG_FLAG==1)
     Serial.println("Sending inference result over BLE..."); // print it to serial consle 
#endif
     inferenceCharacteristic.writeValue(inferenceResult); // Write value
  */    


//#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_ACCELEROMETER
//#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_FUSION
//#error "Invalid model for current sensor"
//#endif
