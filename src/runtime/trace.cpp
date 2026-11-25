#include "tilegen/trace.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace tilegen {
namespace {

struct Event {
    std::string name;
    double duration_us;
    std::thread::id tid;
};

struct Span {
    std::string name;
    std::chrono::steady_clock::time_point start;
};

class Profiler {
  public:
    void begin(const std::string& name) {
        std::lock_guard<std::mutex> lk(mu_);
        spans_.push_back({name, std::chrono::steady_clock::now()});
    }

    void end(const std::string& name) {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto it = spans_.rbegin(); it != spans_.rend(); ++it) {
            if (it->name == name) {
                auto end = std::chrono::steady_clock::now();
                double us = std::chrono::duration_cast<std::chrono::microseconds>(
                                end - it->start)
                                .count();
                events_.push_back({name, us, std::this_thread::get_id()});
                spans_.erase(std::next(it).base());
                return;
            }
        }
    }

    bool flush(const std::string& path) {
        std::string target = path;
        if (target.empty()) {
            const char* env = std::getenv("TILEGEN_PROFILE_OUTPUT");
            if (env == nullptr) return false;
            target = env;
        }
        std::lock_guard<std::mutex> lk(mu_);
        std::ostringstream oss;
        oss << "{\"version\": 1, \"events\": [";
        for (size_t i = 0; i < events_.size(); ++i) {
            if (i) oss << ", ";
            oss << "{\"name\": \"" << events_[i].name << "\", "
                << "\"duration_us\": " << events_[i].duration_us << ", "
                << "\"tid\": " << events_[i].tid << "}";
        }
        oss << "]}\n";
        std::ofstream f(target, std::ios::out | std::ios::trunc);
        if (!f.is_open()) return false;
        f << oss.str();
        return true;
    }

    void reset() {
        std::lock_guard<std::mutex> lk(mu_);
        events_.clear();
        spans_.clear();
    }

  private:
    std::mutex mu_;
    std::vector<Event> events_;
    std::vector<Span> spans_;
};

Profiler& profiler() {
    static Profiler p;
    return p;
}

}  // namespace

void profiler_begin(const std::string& name) { profiler().begin(name); }
void profiler_end(const std::string& name) { profiler().end(name); }
bool profiler_flush(const std::string& path) { return profiler().flush(path); }
void profiler_reset() { profiler().reset(); }

}  // namespace tilegen
