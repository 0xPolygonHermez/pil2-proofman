#ifndef TIMER_GPU_HPP
#define TIMER_GPU_HPP


#ifdef __USE_CUDA__
#include <unordered_map>
#include <string>
#include <cuda_runtime.h>
#include "zklog.hpp"

struct TimerEntry {
    cudaEvent_t start;
    cudaEvent_t stop;
    float timeMs = -1.0f;  // Uninitialized
};

class TimerGPU {
public:
    std::unordered_map<std::string, TimerEntry> timers;
    std::unordered_map<std::string, std::vector<TimerEntry>> multiTimers;
    std::unordered_map<std::string, TimerEntry*> activeCategoryTimers;
    std::vector<std::string> order;
    cudaStream_t stream;

    TimerGPU(cudaStream_t s = 0) : stream(s) {}

    void start(const std::string& name) {
        if (timers.find(name) == timers.end()) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
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
        cudaEventRecord(timers[name].stop, stream);
    }

    void startCategory(const std::string& name) {
        if (activeCategoryTimers.find(name) != activeCategoryTimers.end()) {
            zklog.error("TimerGPU::startCategory called without stop for previous timer: " + name);
            return;
        }
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        multiTimers[name].push_back({start, stop, -1.0f});
        activeCategoryTimers[name] = &multiTimers[name].back();
        cudaEventRecord(start, stream);
    }

    void stopCategory(const std::string& name) {
        auto it = activeCategoryTimers.find(name);
        if (it == activeCategoryTimers.end()) {
            zklog.error("TimerGPU::stopCategory called without matching start: " + name);
            return;
        }

        TimerEntry* entry = it->second;
        cudaEventRecord(entry->stop, stream);
        activeCategoryTimers.erase(it);
    }

    void syncAndCompute(const std::string& name) {
        auto& entry = timers.at(name);
        cudaEventSynchronize(entry.stop);
        cudaEventElapsedTime(&entry.timeMs, entry.start, entry.stop);
    }

    float getTimeMs(const std::string& name) {
        auto& entry = timers.at(name);
        if (entry.timeMs < 0.0f) {
            syncAndCompute(name);
        }
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
            total += entry.timeMs / 1000.0; // Convert ms → s
        }

        return total;
    }

    void syncAndLogAll() {
        for (const auto& name : order) {
            auto& entry = timers[name];
            if (entry.timeMs < 0.0f) {
                cudaEventSynchronize(entry.stop);
                cudaEventElapsedTime(&entry.timeMs, entry.start, entry.stop);
            }
            zklog.trace("<-- " + name + " : " + std::to_string(entry.timeMs / 1000.0f) + " s");
        }
    }

    void syncCategories() {
        for (auto& [category, entries] : multiTimers) {
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
        double time_total = getTimeSec(total_name);

        if (multiTimers.empty()) return;

        zklog.trace("     KERNELS CONTRIBUTIONS:");

        std::vector<std::pair<std::string, double>> category_times;
        double accounted_time = 0.0;

        // Collect and compute total times
        for (const auto& [category, entries] : multiTimers) {
            double total_sec = getCategoryTotalTimeSec(category);
            accounted_time += total_sec;
            category_times.emplace_back(category, total_sec);
        }

        // Sort by descending time
        std::sort(category_times.begin(), category_times.end(),
                [](const auto& a, const auto& b) {
                    return a.second > b.second;
                });

        // Format and log
        std::ostringstream oss;
        for (const auto& [category, total_sec] : category_times) {
            oss << std::fixed << std::setprecision(2) << total_sec << "s (";
            oss << std::setprecision(2) << (total_sec / time_total) * 100.0 << "%";
            oss << ")";
            zklog.trace("        " + category + std::string(15 - std::min<size_t>(15, category.size()), ' ') + ":  " + oss.str());
            oss.str("");
            oss.clear();
        }

        double other_time = time_total - accounted_time;
        if (other_time < 0.0) other_time = 0.0;

        oss << std::fixed << std::setprecision(2) << other_time << "s (";
        oss << std::setprecision(2) << (other_time / time_total) * 100.0 << "%";
        oss << ")";
        zklog.trace("        OTHER" + std::string(15 - 5, ' ') + ":  " + oss.str());
    }

    ~TimerGPU() {
        for (auto& [_, entry] : timers) {
            cudaEventDestroy(entry.start);
            cudaEventDestroy(entry.stop);
        }
    }
};

inline std::string makeTimerName(const std::string& base, int id) {
    return base + "_" + to_string(id);
}

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

#define TimerSyncAndLogAllGPU(timer) (timer.syncAndLogAll())

#define TimerSyncCategoriesGPU(timer) (timer.syncCategories())

#define TimerResetGPU(timer) (timer.clear())

#define TimerGetElapsedCategoryGPU(timer, category) \
    (timer.getCategoryTotalTimeSec(#category))

#define TimerLogCategoryContributionsGPU(timer, total_name) \
    (timer.logCategoryContributions(#total_name))

#endif // TIMER_GPU_HPP
#endif