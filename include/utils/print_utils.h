#ifndef INCLUDE_LATTICE_PRINT_UTILS_H
#define INCLUDE_LATTICE_PRINT_UTILS_H

namespace MG {
    /*! some log levels */
    enum LogLevel { ERROR = 0, INFO, DEBUG, DEBUG2, DEBUG3 };

    /**
     * 	Log only shown by the primary node
     * 	\param level: if the level is <= the current log level, a message will be printed
     * 	\param format: printf string format
     * 	\param variable list of arguments
     *
     * 	If level is ERROR, abort the process after showing the error.
     */
    void MasterLog(LogLevel level, const char *, ...);

    /*! All Nodes Perform Logging */
    void LocalLog(LogLevel level, const char *, ...);

    /*! Set the log level */
    void SetLogLevel(LogLevel level);

    /*! Get the current log level */
    LogLevel GetLogLevel(void);

    /**
     * Abort if no progress is set after monitor_time seconds
     * \param monitor_time: maximum seconds without progress
     *
     * If monitor_time <= 0, then no monitoring is performed.
     */
    void SetMonitorTime(int monitor_time);

    /**
     * Pause monitoring progress
     */

    void PauseMonitor();

    /**
     * Set progress
     */

    void SetProgress();
}

#endif
