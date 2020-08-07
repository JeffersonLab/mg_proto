/*
 * timer.h
 *
 *  Created on: Sep 12, 2018
 *      Author: tkurth
 */

#ifndef INCLUDE_UTILS_TIMER_H_
#define INCLUDE_UTILS_TIMER_H_

#include "MG_config.h"
#include "utils/print_utils.h"
#include <assert.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string.h>
#include <unordered_map>

namespace MG {

    namespace Timer {

        class Timer {
        public:
            Timer() { Reset(); };

            void Reset() {
                tDeltaTotal = std::chrono::duration<double>::zero();
                isStarted = false;
                num_calls = 0;
            };

            void Start() {
                isStarted = true;
                tStart = std::chrono::high_resolution_clock::now();
            };

            void Stop() {
                tEnd = std::chrono::high_resolution_clock::now();
                assert(isStarted == true);
                auto dur = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart);
                //update total
                tDeltaTotal += dur;
                //reset started
                isStarted = false;
                num_calls++;
            };

            double getTotalDuration() const { return tDeltaTotal.count(); }

            double getAvgDuration() const { return tDeltaTotal.count() / num_calls; }

        private:
            std::chrono::high_resolution_clock::time_point tStart, tEnd;
            bool isStarted;
            std::chrono::duration<double> tDeltaTotal;
            std::uint64_t num_calls;
        };

        class TimerAPI {
        public:
            static void addTimer(const std::string &key) {
#ifdef MG_ENABLE_TIMERS
                timers[key] = Timer();
#else
                (void)key;
#endif
            }

            static void startTimer(const std::string &key) {
#ifdef MG_ENABLE_TIMERS
                timers[key].Start();
#else
                (void)key;
#endif
            }

            static void stopTimer(const std::string &key) {
#ifdef MG_ENABLE_TIMERS
                timers[key].Stop();
#else
                (void)key;
#endif
            }

            static void resetTimer(const std::string &key) {
#ifdef MG_ENABLE_TIMERS
                timers[key].Reset();
#else
                (void)key;
#endif
            }

            static void reset() { timers.clear(); }

            static void reportAllTimer() {
                for (auto const &item : timers) {
                    std::string key = item.first;
                    auto value = item.second;
                    MasterLog(INFO, "TIMING %s = %lf s  avg per call %1f", key.c_str(),
                              value.getTotalDuration(), value.getAvgDuration());
                }
            }

            static std::unordered_map<std::string, Timer> timers;

        private:
            TimerAPI() {}
        };
    }
}

#endif
