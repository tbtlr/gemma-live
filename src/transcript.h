// Text helpers for the rolling ASR transcript.
//
// Both operate on the " word word word " shape normalise() produces: a single
// leading and trailing space, lowercase, punctuation collapsed to spaces. The
// padding is what makes substring matching land on word boundaries, so
// " gemma " cannot match inside "gemmalike".
#pragma once

#include <cctype>
#include <set>
#include <string>

namespace transcript {

// Lowercase, keep only alphanumerics, collapse everything else to single
// spaces, and pad both ends with one space.
inline std::string normalise(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back(' ');
    bool prev_space = true;
    for (char c : s) {
        const unsigned char u = (unsigned char) c;
        char ch;
        if (std::isalpha(u) || std::isdigit(u)) {
            ch = (char) std::tolower(u);
            prev_space = false;
        } else {
            ch = ' ';
            if (prev_space) continue;
            prev_space = true;
        }
        out.push_back(ch);
    }
    if (out.empty() || out.back() != ' ') out.push_back(' ');
    return out;
}

// The last word of a normalised transcript, or "" if there isn't one.
inline std::string last_word(const std::string & norm) {
    const size_t end = norm.find_last_not_of(' ');
    if (end == std::string::npos) return {};
    const size_t beg = norm.find_last_of(' ', end);
    if (beg == std::string::npos) return norm.substr(0, end + 1);
    return norm.substr(beg + 1, end - beg);
}

// True when the transcript's last word is one an English sentence essentially
// cannot end on — the speaker is mid-thought, and the pause we are hearing is
// them thinking rather than finishing.
//
// Only words where continuation is near-certain are listed. The rolling
// transcript is noisy (it rewrites and hallucinates constantly), but a garbled
// last word simply is not in the set and the caller falls back to its default
// — the failure mode is "no change", never a wrong call in the costly
// direction.
inline bool ends_mid_thought(const std::string & norm) {
    static const std::set<std::string> hanging = {
        // conjunctions / subordinators
        "and", "but", "or", "so", "because", "although", "though", "while",
        "if", "when", "whether", "unless", "since", "that", "than", "as",
        // prepositions
        "to", "for", "with", "about", "from", "of", "in", "on", "at", "by",
        "into", "onto", "over", "under", "between", "like",
        // determiners
        "the", "a", "an", "my", "your", "his", "her", "its", "our", "their",
        "this", "these", "those", "some", "any", "every",
        // auxiliaries / copulas
        "is", "are", "was", "were", "am", "be", "been", "being", "do",
        "does", "did", "have", "has", "had", "will", "would", "can",
        "could", "should", "might", "must",
    };
    return hanging.count(last_word(norm)) > 0;
}

}  // namespace transcript
