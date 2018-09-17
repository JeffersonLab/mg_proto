/*
 * timer.h
 *
 *  Created on: Sep 12, 2018
 *      Author: tkurth
 */

#ifndef INCLUDE_UTILS_TIMER_H_
#define INCLUDE_UTILS_TIMER_H_

#include <cstdlib>
#include <chrono>
#include <assert.h>
#include <string.h>
#include <unordered_map>
#include <iostream>
#include <memory>
#include "utils/print_utils.h"

namespace MG {
    
    namespace Timer {
        
        class Timer {
        public:
            Timer(){
                Reset();
            };
            
            void Reset(){
                tDeltaTotal = std::chrono::duration<double>(0);
                tDeltaMin = std::chrono::duration<double>(0);
                tDeltaMax = std::chrono::duration<double>(0);
                isStarted = false;
            };
            
            void Start(){ 
                isStarted = true;
                tStart = std::chrono::high_resolution_clock::now(); 
            };
            
            void Stop(){
                tEnd = std::chrono::high_resolution_clock::now(); 
                assert(isStarted == true);
                auto dur = std::chrono::duration_cast< std::chrono::duration<double> >(tEnd - tStart);
                //update total
                tDeltaTotal += dur;
                //reset started
                isStarted = false;
            };
            
            std::chrono::duration<double> getTotalDuration() const{
                return tDeltaTotal;
            }
            
        private:
            std::chrono::high_resolution_clock::time_point tStart, tEnd;
            bool isStarted;
            std::chrono::duration<double> tDeltaTotal, tDeltaMin, tDeltaMax;
        };
        
        class TimerAPI {
        public:
            static std::shared_ptr<TimerAPI> getInstance(){
                static std::shared_ptr<TimerAPI> instance{new TimerAPI };
                return instance;
            };
            
            void addTimer(const std::string& key){
                timers[key] = Timer::Timer();
            };
            
            void startTimer(const std::string& key){
                timers[key].Start();
            };
            
            void stopTimer(const std::string& key){
                timers[key].Stop();
            };
            
            void resetTimer(const std::string& key){
                timers[key].Reset();
            };
            
            void reportAllTimer() const{
                for(auto const &item : timers) {
                    std::string key = item.first;
                    auto value = item.second;
                    // std::cout << "INFO: TIMING " << key << " = " << value.getTotalDuration().count() << std::endl;
                    MasterLog(INFO, "TIMING %s = %lf (sec.)", key.c_str(), value.getTotalDuration().count());

                }
            }
            
            //delete some operators
            TimerAPI(TimerAPI const&) = delete;
            void operator=(TimerAPI const&) = delete;
            
        private:
            TimerAPI() {

            	//MasterLog(INFO, "Constructing TimerAPI");
            }
            std::unordered_map<std::string, Timer> timers;
        };
    }
}

#endif
