/***
*	Powermon Control Library
*
*	Michele Scala
*	Federico Busato
*	University of Verona, Italy, Dept. of Computer Science, 10 november 2015
*/
#include <exception>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <string.h>

#include "powermon.hpp"

const double Powermon::V_FULLSCALE = 26.52;
const double Powermon::R_SENSE = 0.00422;    // found empirically
const double Powermon::I_FULLSCALE = 0.10584;

Powermon::Powermon() : device("/dev/ttyUSB0"),  mask(22u), log_enabled(false), log_stream(NULL) {}


Powermon::Powermon(std::ofstream* _log) :
                        device("/dev/ttyUSB0"),  mask(22u), log_enabled(true), log_stream(_log) {}

Powermon::Powermon(const char* _device, uint16_t _mask) :
                        device(_device),  mask(_mask), log_enabled(false), log_stream(NULL) {}

Powermon::Powermon(const char* _device, uint16_t _mask, std::ofstream* _log) :
                        device(_device),  mask(_mask), log_enabled(true), log_stream(_log) {}

//------------------------------------------------------------------------------

bool Powermon::configure() {
    this->pw_file_descriptor = ::open(device, O_RDWR | O_NOCTTY);
	if (this->pw_file_descriptor < 0)
        //throw std::runtime_error("Powermon::Configure : TTY not found");
        return 0;
	if ((this->pw_file_pointer = ::fdopen(pw_file_descriptor, "r+")) == NULL)
        throw std::runtime_error("Powermon::Configure : Can't open TTY");

	::tcgetattr(this->pw_file_descriptor, &pw_old_config); /* save current port settings */
	::bzero(&pw_config, sizeof (pw_config));

    /*
    CRTSCTS == hardware flow control
    CS8 = 8 bits
    CLOCAL = ignore modem control lines
    CREAD = enable receiver
    B38400 = baud rate
    */
	/* configure port */
	this->pw_config.c_iflag = IGNPAR;
	this->pw_config.c_cflag = BAUDRATE | CS8 | CSTOPB | CLOCAL | CREAD;
	this->pw_config.c_oflag = 0;

	/* set input mode (non-canonical, no echo,...) */
	this->pw_config.c_lflag = 0;

	this->pw_config.c_cc[VTIME] = 0; /* inter-character timer unused */
	this->pw_config.c_cc[VMIN] = 1; /* blocking read until 1 char received */

	::tcflush(this->pw_file_descriptor, TCIFLUSH);
	::tcsetattr(this->pw_file_descriptor, TCSANOW, &pw_config);
    return 1;
}

void Powermon::init() {
	char buffer[32];

	::fprintf(this->pw_file_pointer, "d\n");
	::usleep(100000);
	::fflush(this->pw_file_pointer);
	::usleep(100000);
	::tcflush(this->pw_file_descriptor, TCIFLUSH);
	::fprintf(this->pw_file_pointer, "\n");
	//get result
	::fgets(buffer, sizeof(buffer), this->pw_file_pointer);
    if (::strcmp(buffer, "OK\r\n") != 0)
    	throw std::runtime_error("Powermon::Init");
}

void Powermon::setMask() {
	char buffer[32];

	unsigned length = ::sprintf(buffer, "m %u\n", this->mask);
	::fwrite(buffer, 1, length, this->pw_file_pointer);
	//get response
	::fgets(buffer, sizeof(buffer), this->pw_file_pointer);

    unsigned setval;
	::sscanf(buffer, "M=%u\r", &setval);
	//get result
	::fgets(buffer, sizeof(buffer), this->pw_file_pointer);
    if (::strcmp(buffer, "OK\r\n") != 0)
    	throw std::runtime_error("Powermon::setMask");
}

void Powermon::setSamples() {
	char buffer[32];

	unsigned length = ::sprintf(buffer, "s %u %u\n", SAMPLING_INTERVALL, N_OF_SAMPLING);
	::fwrite(buffer, 1, length, this->pw_file_pointer);
	::fgets(buffer, sizeof(buffer), this->pw_file_pointer);

    unsigned set_interval, set_num_samples;
	::sscanf(buffer, "S=%u,%u\r", &set_interval, &set_num_samples);

	::fgets(buffer, sizeof(buffer), this->pw_file_pointer);
    if (::strcmp(buffer, "OK\r\n") != 0)
    	throw std::runtime_error("Powermon::setSamples");
}

void Powermon::setTime() {
	char buffer[32];

	unsigned length = ::sprintf(buffer, "t %u\n", 1);
	::fwrite(buffer, 1, length, this->pw_file_pointer);
	::fgets(buffer, sizeof(buffer), this->pw_file_pointer);

    unsigned setval;
	::sscanf(buffer, "T=%u\r", &setval);

	::fgets(buffer, sizeof(buffer), this->pw_file_pointer);
    if (::strcmp(buffer, "OK\r\n") != 0)
    	throw std::runtime_error("Powermon::setTime");
}

//------------------------------------------------------------------------------

void Powermon::getVersion() {
	char buffer[128];
	int length;

	length = sprintf(buffer, "v\n");
	fwrite(buffer, 1, length, pw_file_pointer);

	//get response
	fgets(buffer, sizeof(buffer), pw_file_pointer);
	std::cout << "Version: " << buffer << std::endl;

	fgets(buffer, sizeof(buffer), pw_file_pointer);
	std::cout << buffer << std::endl;

	fgets(buffer, sizeof(buffer), pw_file_pointer);
	std::cout << buffer << std::endl;

	//get result
	fgets(buffer, sizeof(buffer), pw_file_pointer);
    if (::strcmp(buffer, "OK\r\n") != 0)
    	throw std::runtime_error("Powermon::getVersion");
}

//------------------------------------------------------------------------------


void Powermon::start() {
    this->power_max = 0;
    this->power_total = 0;
    this->sampled_instants = 0;

    this->stop_task = false;
    this->first_read = false;


    if (this->configure()) {
        this->power_max = 0;
        this->power_total = 0;
        this->sampled_instants = 0;

        this->stop_task = false;
        this->first_read = false;
        this->init();
        this->setMask();
        this->setSamples();
        this->setTime();

    	supportThread = new std::thread(Powermon::task, this);
    	while(!first_read);
    } else {
        this->power_max = NAN;
        this->power_total = NAN;
        this->sampled_instants = NAN;
    }
}

void Powermon::stop() {
    if (!powermon_found)
        return;
	stop_task = true;
    (*supportThread).join();
    delete supportThread;
}

int Powermon::getSampledInstants() {
	return this->sampled_instants;
}

double Powermon::getPowerMax() {
	return this->power_max;
}

double Powermon::getPowerAvg() {
	return (double) this->power_total / this->sampled_instants;
}

//------------------------------------------------------------------------------

void Powermon::task(void* ptr){
	Powermon* master = (Powermon*) ptr;
	master->mutex.lock();

	::fprintf(master->pw_file_pointer, "e\n");

	int times[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    std::stringstream ss_mod3;
    int sensor_index = 0;
    double power_sensor[ACTIVE_SENSORS];

	while (true) {
        unsigned char buffer[4];
		::fread(buffer, 1, 4, master->pw_file_pointer);
		uint8_t flags = buffer[0];

		if (!(flags & (1<<TIME_FLAG)) && (flags & (1<<DONE_FLAG))) break;
		if(flags & (1 << OVF_FLAG))
            std::cerr << "*" << std::endl;
		if (flags & (1 << TIME_FLAG)) {
			unsigned int time = (((uint32_t)buffer[0] << 24) | ((uint32_t)buffer[1] << 16)
            					| ((uint32_t)buffer[2] << 8) | ((uint32_t)buffer[3]))
            					& 0x3FFFFFFF;
            std::cerr << "timestamp" << time <<std::endl;
		} else {
			master->first_read = true;

			uint8_t sensor = (uint8_t) buffer[0] & 0x0F;
			uint16_t voltage = ((uint16_t) buffer[1] << 4) | ((buffer[3] >> 4) & 0x0F);
			uint16_t current = ((uint16_t) buffer[2] << 4) | (buffer[3] & 0x0F);
			double v_double = (double) voltage / 4096 * V_FULLSCALE;
			double i_double = (double) current / 4096 * I_FULLSCALE / R_SENSE;

			double watt = v_double * i_double;
            power_sensor[sensor_index] = watt;

			if (master->log_enabled) {
                ss_mod3 << times[sensor] << "\t" << (int) sensor
                                         << "\t" << v_double << "\t" << i_double
                                         << "\t" << watt << std::endl;
            }
			times[sensor]++;

            sensor_index++;
            if (sensor_index == ACTIVE_SENSORS) {
                sensor_index = 0;
                if (master->log_enabled){
                    (*(master->log_stream)) << ss_mod3.rdbuf();}

                int sum = std::accumulate(power_sensor, power_sensor + ACTIVE_SENSORS, 0);
                master->power_total += sum;
                if (sum > master->power_max)
                    master->power_max = sum;

                master->sampled_instants++;
            }

			//STOP READ ACTIVE_SENSORS
			if (master->stop_task) break;
		}
	}

	//master->init(); <<--------- perchè?
	::tcflush(master->pw_file_descriptor, TCIOFLUSH);
	::tcsetattr(master->pw_file_descriptor, TCSANOW, &master->pw_old_config);
	::close(master->pw_file_descriptor);

	master->mutex.unlock();
}
