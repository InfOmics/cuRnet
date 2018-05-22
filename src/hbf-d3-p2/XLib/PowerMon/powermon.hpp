/***
*	Powermon Control Library
*
*	Michele Scala
*	Federico Busato
*	University of Verona, Italy, Dept. of Computer Science, 10 november 2015
*/
#pragma once

#include <fstream>
#include <thread>
#include <mutex>

#include <stdint.h>
#include <sys/termios.h>
#include <time.h>
#include <stdio.h>

#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>

class Powermon {

static const int VERSION_H = 2;
static const int VERSION_L = 2;
static const int OVF_FLAG = 7;
static const int TIME_FLAG = 6;
static const int DONE_FLAG = 5;
static const int ACTIVE_SENSORS = 3;

#define BAUDRATE B1000000
static const double V_FULLSCALE;
static const double R_SENSE;    // found empirically
static const double I_FULLSCALE;

public:

    /**
     * @brief Create the Powermoon object.
     * @details
     *
     * @param p Name of the device port ( e.g. /dev/ttyUSB0 ).
     * @param m Bit Mask of sensors, every 1 bit enable a sensor.
    */
    Powermon();
    Powermon(std::ofstream* _log);
    Powermon(const char* _device, uint16_t _mask);
    Powermon(const char* _device, uint16_t _mask, std::ofstream* _log);

	/**
	 * @brief Start getting samples from the powermoon.
	 * @details Start getting samples from the powermoon, it's a blocking function, it waits that powermoon starts sending samples before return.
	 *
	 * @param frequency Sampling frequency
	 * @param samples Number of samples to take.
	 *
	 * @return 1 if something got wrong
	 */
	void start();
	/**
	 * @brief Stops getting samples if it is running.
	 * @details Stops getting samples if it is running.
	 */
	void stop();
	/**
	 * @brief Waits the end of getting samples execution.
	 * @details Waits the end of getting samples execution.
	 */
	void wait();

	/**
	 * @brief Sets the output stream where logs will be written.
	 * @details Sets the output stream where logs will be written.
	 *
	 * @param out Log output stream.
	 * @param divider String that divides every information in a line.
	 * @param time Logs with time?
	 */
	//void log(std::ostream* out,bool time);

    double getPowerMax();
    double getPowerAvg();
    int getSampledInstants();

private:
    const int SAMPLING_INTERVALL = 1;
    const int N_OF_SAMPLING = 30000;

    double power_max;
    double power_total;
    int sampled_instants;
    bool powermon_found;
	/**
	 * @brief Configure the tty, connect and open it
	 * @return 0 all ok, !=0 if there is some errors
	 */
	bool configure();

    void init();
	/**
	 * @brief Set sensors mask to powermoon
	 * @return 0 all ok, !=0 if there is some errors
	 */
	void setMask();
	/**
	 * @brief Set the number of samples to powermoon
	 * @return 0 all ok, !=0 if there is some errors
	 */
	void setSamples();
	/**
	 * @brief Set the current time to powermoon
	 * @return 0 all ok, !=0 if there is some errors
	 */
	void setTime();
	/**
	 * @brief Reset the powermoon
	 * @details Reset the powermoon, mutex necessary if thread is running
	 * @return 0 all ok, !=0 if there is some errors
	 */
	void reset();

	void getVersion();

	//PARAMS
	const char*	device;
	const unsigned short mask;
    const bool log_enabled;
    std::ostream*	log_stream;

	//THREAD
	std::mutex 	   mutex;
	std::thread*   supportThread;
	static void    task(void* point);
	bool 		   stop_task;
	bool		   first_read;

	//SERIAL
	int 		pw_file_descriptor;
	FILE* 		pw_file_pointer;

	//CONFIG
    struct termios pw_config, pw_old_config;
};
