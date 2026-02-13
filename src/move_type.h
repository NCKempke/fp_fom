#pragma once


enum class move_type {
    random = 0,
    oneopt,
    oneopt_greedy,
    flip,
    mtm_unsat,
    mtm_sat
};

inline const char* toString(const move_type m) {
    switch (m) {
        case move_type::random:
            return "random";
        case move_type::oneopt:
            return "oneopt";
        case move_type::oneopt_greedy:
            return "oneopt_greedy";
        case move_type::flip:
            return "flip";
        case move_type::mtm_unsat:
            return "mtm_unsat";
        case move_type::mtm_sat:
            return "mtm_sat";
        default:
            return "unknown";
    }
}
