#include "utils/print_utils.h"
#include "MG_config.h"
#include "utils/initialize.h"
#include <chrono>
#include <cstdarg>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#ifdef MG_QMP_COMMS
#    include "qmp.h"
#endif

namespace MG {

    /* Current Log Level */
    static volatile LogLevel current_log_level = MG_DEFAULT_LOGLEVEL;

    /* An array holding strings corresponding to log levels */
    static std::string log_string_array[] = {"ERROR", "INFO", "DEBUG", "DEBUG2", "DEBUG3"};

    /**
     * SetLogLevel -- set the log level
     *
     * \param level  -- The LogLevel to set. LogMessages with levels <= level will be printed
     *
     * NB: This function may be called in several MPI processes, in which case it needs to
     * be called collectively. Being called in one MPI process and not in another is considered
     * a programming error. Likewise the function may be concurrently called from several
     * OpenMP threads. If called from many threads potentially. While it would be weird for
     * all threads to set different log levels, the safe thing to do is to guard the write with
     * an OMP Criticla section
     *
     */
    void SetLogLevel(LogLevel level) {
#pragma omp master
        current_log_level = level;
#pragma omp barrier
    }

    /**
     * SetLogLevel - get the log level
     *
     * \returns  The current log level
     *
     * NB: The design is for the loglevel to be kept on each MPI process. This function only
     * reads the loglevel value, so no races can occur.
     */
    LogLevel GetLogLevel(void) { return current_log_level; }

    /**
     * 	LocalLog - Local process performs logging
     * 	\param level -- if the level is <= the current log level, a message will be printed
     * 	\param format_string
     * 	\param variable list of arguments
     *
     * 	Current definition is that only the master thread on each nodes logs
     */
    void LocalLog(LogLevel level, const char *format, ...) {
        va_list args;
        va_start(args, format);
#pragma omp master
        {
            if (level <= current_log_level) {
#ifdef MG_QMP_COMMS
                int size = QMP_get_number_of_nodes();
                int rank = QMP_get_node_number();
#else
                int size = 1;
                int rank = 0;
#endif
                printf("%s: Rank %d of %d: ", log_string_array[level].c_str(), rank, size);

                vprintf(format, args);

                printf("\n");
            }
            va_end(args);
            /* If the level is error than we should abort */
            if (level == ERROR) { MG::abort(); }
        }
    }

    class Monitor : public std::enable_shared_from_this<Monitor> {
    public:
        volatile int interval;        // Check the state each this number of seconds
        volatile bool state;          // if false, indicates progress
        volatile bool paused;         // if the monitoring is paused
        std::shared_ptr<Monitor> own; // Reference to this (set if thread is running)

        Monitor(int interval_) : interval(interval_), state(false), paused(true) {}
        Monitor() : Monitor(0) {}

        // Function run by the thread
        void operator()() {
            while (interval > 0) {
                if (state && !paused) {
                    MasterLog(ERROR, "No progress after %d seconds", interval);
                    break;
                }
                state = true;
                std::this_thread::sleep_for(std::chrono::seconds(interval));
            }
            own.reset();
        }

        void Set() {
            // Set progress
            state = false;
            paused = false;

            // Create a thread to monitor a progress if necessary
            if (interval > 0 && !paused && !own) {
                // Avoid destruction of this object while the thread is running by creating
                // another reference to this objected
                own = shared_from_this();

                // Create a detached thread
                std::thread t = std::thread(&Monitor::operator(), this);
                t.detach();
            }
        }

        void Pause() { paused = true; }

        void Finish() { interval = 0; }
    };

    static std::shared_ptr<Monitor> monitor;

    /**
     * 	Log only shown by the primary node
     * 	\param level: if the level is <= the current log level, a message will be printed
     * 	\param format: printf string format
     * 	\param variable list of arguments
     *
     * 	If level is ERROR, abort the process after showing the error.
     */
    void MasterLog(LogLevel level, const char *format, ...) {
        va_list args;
        va_start(args, format);
#pragma omp master
        {

#ifdef MG_QMP_COMMS
            if (QMP_is_primary_node())
#endif
            {
                if (level <= current_log_level) {
                    printf("%s: ", log_string_array[level].c_str());
                    vprintf(format, args);
                    printf("\n");
                }

                /* If the level is error than we should abort */
                if (level == ERROR) { MG::abort(); }

                // Set progress
                SetProgress();
            }
        }
        va_end(args);
    }

    /**
     * Abort if no progress is set after monitor_time seconds
     * \param monitor_time: maximum seconds without progress
     *
     * If monitor_time <= 0, then no monitoring is performed.
     */

    void SetMonitorTime(int monitor_time) {
#ifdef MG_QMP_COMMS
        if (QMP_is_primary_node())
#endif
        {
            if (monitor_time > 0 && (!monitor || monitor->interval != monitor_time))
                monitor = std::make_shared<Monitor>(monitor_time);

            if (monitor_time <= 0 && monitor) {
                monitor->Finish();
                monitor.reset();
            }
        }
    }

    /**
     * Pause monitoring progress
     */

    void PauseMonitor() {
        if (monitor) monitor->Pause();
    }

    /**
     * Set progress
     */

    void SetProgress() {
        if (monitor) monitor->Set();
    }
}
