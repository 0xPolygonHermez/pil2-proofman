#ifndef TIMER_GPU_HPP
#define TIMER_GPU_HPP


#include <unordered_map>
#include <string>
#include <cuda_runtime.h>
#include "zklog.hpp"

#define LOG_TIME_GPU 1

struct TimerEntry {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    float timeMs = -1.0f;
};

class TimerGPU {
public:
    std::unordered_map<std::string, TimerEntry> timers;
    std::unordered_map<std::string, std::vector<TimerEntry>> multiTimers;
    std::unordered_map<std::string, TimerEntry*> activeCategoryTimers;
    std::vector<std::string> order;
    cudaStream_t stream = nullptr;

    TimerGPU() = default;
    explicit TimerGPU(cudaStream_t s) : stream(s) {}

    void init(cudaStream_t s) { stream = s; }

    bool createEvent(cudaEvent_t& event) {
        cudaError_t err = cudaEventCreate(&event);
        if (err != cudaSuccess) {
            zklog.error("cudaEventCreate failed: " + std::string(cudaGetErrorString(err)));
            return false;
        }
        return true;
    }

    void start(const std::string& name) {
        if (timers.find(name) == timers.end()) {
            cudaEvent_t start, stop;
            if (!createEvent(start) || !createEvent(stop)) return;
            timers[name] = {start, stop, -1.0f};
            order.push_back(name);
        }
        cudaEventRecord(timers[name].start, stream);
    }

    void stop(const std::string& name) {
        auto it = timers.find(name);
        if (it == timers.end()) {
            zklog.error("TimerGPU::stop called for unknown section: " + name);
            return;
        }
        cudaEventRecord(it->second.stop, stream);
    }

    void startCategory(const std::string& name) {
        if (activeCategoryTimers.find(name) != activeCategoryTimers.end()) {
            zklog.error("TimerGPU::startCategory called without stop for previous timer: " + name);
            return;
        }

        cudaEvent_t start, stop;
        if (!createEvent(start) || !createEvent(stop)) return;

        multiTimers[name].emplace_back(TimerEntry{start, stop, -1.0f});
        TimerEntry& entry = multiTimers[name].back();

        activeCategoryTimers[name] = &entry;
        cudaEventRecord(entry.start, stream);
    }

    void stopCategory(const std::string& name) {
        auto it = activeCategoryTimers.find(name);
        if (it == activeCategoryTimers.end()) {
            zklog.error("TimerGPU::stopCategory called without matching start: " + name);
            return;
        }
        cudaEventRecord(it->second->stop, stream);
        activeCategoryTimers.erase(it);
    }

    void syncAndCompute(const std::string& name) {
        auto& entry = timers.at(name);
        cudaEventSynchronize(entry.stop);
        cudaEventElapsedTime(&entry.timeMs, entry.start, entry.stop);
    }

    float getTimeMs(const std::string& name) {
        auto& entry = timers.at(name);
        if (entry.timeMs < 0.0f) syncAndCompute(name);
        return entry.timeMs;
    }

    double getTimeSec(const std::string& name) {
        return getTimeMs(name) / 1000.0;
    }

    double getCategoryTotalTimeSec(const std::string& category) {
        double total = 0.0;
        auto it = multiTimers.find(category);
        if (it == multiTimers.end()) return 0.0;

        for (auto& entry : it->second) {
            if (entry.timeMs < 0.0f) {
                cudaEventSynchronize(entry.stop);
                cudaEventElapsedTime(&entry.timeMs, entry.start, entry.stop);
            }
            total += entry.timeMs / 1000.0;
        }
        return total;
    }

    void syncAndLogAll(std::string instance_id, std::string airgroup_id, std::string air_id) {
        zklog.trace("TIMERS FOR INSTANCE ID " + instance_id + " [" + airgroup_id + ":" + air_id + "]");
        for (const auto& name : order) {
            auto& entry = timers[name];
            if (entry.timeMs < 0.0f) syncAndCompute(name);
            zklog.trace("<-- " + name + " : " + std::to_string(entry.timeMs / 1000.0f) + " s");
        }
    }

    void syncCategories() {
        for (auto& [_, entries] : multiTimers) {
            for (auto& entry : entries) {
                if (entry.timeMs < 0.0f) {
                    cudaEventSynchronize(entry.stop);
                    cudaEventElapsedTime(&entry.timeMs, entry.start, entry.stop);
                }
            }
        }
    }

    void clear() {
        for (auto& [_, entry] : timers) {
            cudaEventDestroy(entry.start);
            cudaEventDestroy(entry.stop);
        }
        timers.clear();
        order.clear();

        for (auto& [_, entries] : multiTimers) {
            for (auto& entry : entries) {
                cudaEventDestroy(entry.start);
                cudaEventDestroy(entry.stop);
            }
        }
        multiTimers.clear();
        activeCategoryTimers.clear();
    }

    void logCategoryContributions(const std::string& total_name) {
        if (timers.find(total_name) == timers.end()) return;

        double time_total = getTimeSec(total_name);
        if (multiTimers.empty()) return;

        zklog.trace("     KERNELS CONTRIBUTIONS:");

        std::vector<std::pair<std::string, double>> category_times;
        double accounted_time = 0.0;

        for (const auto& [category, entries] : multiTimers) {
            double total_sec = getCategoryTotalTimeSec(category);
            accounted_time += total_sec;
            category_times.emplace_back(category, total_sec);
        }

        std::sort(category_times.begin(), category_times.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        std::ostringstream oss;
        for (const auto& [category, total_sec] : category_times) {
           oss << std::fixed << std::setprecision(4) << total_sec << "s (" << std::setprecision(2) << (total_sec / time_total) * 100.0 << "%)";
            zklog.trace("        " + category + std::string(15 - std::min<size_t>(15, category.size()), ' ') + ":  " + oss.str());
            oss.str("");
            oss.clear();
        }

        double other_time = std::max(0.0, time_total - accounted_time);
        oss << std::fixed << std::setprecision(4) << other_time << "s (" << std::setprecision(2)<< (other_time / time_total) * 100.0 << "%)";
        zklog.trace("        OTHER" + std::string(15 - 5, ' ') + ":  " + oss.str());
    }

    ~TimerGPU() {
        clear();
    }
};

inline std::string makeTimerName(const std::string& base, int id) {
    return base + "_" + std::to_string(id);
}

#if LOG_TIME_GPU
#define TimerStartIdGPU(timer, name, id) \
    timer.start(makeTimerName(#name, id)); \

#define TimerStopIdGPU(timer, name, id) \
    timer.stop(makeTimerName(#name, id))

#define TimerStartCategoryGPU(timer, category) \
    timer.startCategory(#category); \

#define TimerStopCategoryGPU(timer, category) \
    timer.stopCategory(#category); \

#define TimerStartGPU(timer, name) timer.start(#name);
#define TimerStopGPU(timer, name)  timer.stop(#name)
#define TimerStopAndLogGPU(timer, name) \
    timer.stop(#name); \
    timer.syncAndCompute(#name); \
    zklog.trace("<-- " #name " : " + std::to_string(timer.getTimeMs(#name) / 1000.0f) + " s")

#define TimerGetElapsedGPU(timer, name) (timer.getTimeSec(#name))

#define TimerSyncAndLogAllGPU(timer, instance_id, airgroup_id, air_id) (timer.syncAndLogAll(std::to_string(instance_id), std::to_string(airgroup_id), std::to_string(air_id)))

#define TimerSyncCategoriesGPU(timer) (timer.syncCategories())

#define TimerResetGPU(timer) (timer.clear())

#define TimerGetElapsedCategoryGPU(timer, category) \
    (timer.getCategoryTotalTimeSec(#category))

#define TimerLogCategoryContributionsGPU(timer, total_name) \
    (timer.logCategoryContributions(#total_name))

#else 
#define TimerStartIdGPU(timer, name, id)
#define TimerStopIdGPU(timer, name, id)
#define TimerStartCategoryGPU(timer, category)
#define TimerStopCategoryGPU(timer, category)
#define TimerStartGPU(timer, name)
#define TimerStopGPU(timer, name)
#define TimerStopAndLogGPU(timer, name)
#define TimerGetElapsedGPU(timer, name) 0.0
#define TimerSyncAndLogAllGPU(timer, instance_id, airgroup_id, air_id)
#define TimerSyncCategoriesGPU(timer)
#define TimerResetGPU(timer)
#define TimerGetElapsedCategoryGPU(timer, category) 0.0
#define TimerLogCategoryContributionsGPU(timer, total_name)
#define TimerSyncCategoriesGPU(timer)
#define TimerResetGPU(timer)
#define TimerGetElapsedCategoryGPU(timer, category) 0.0
#define TimerLogCategoryContributionsGPU(timer, total_name)
#endif
#endif