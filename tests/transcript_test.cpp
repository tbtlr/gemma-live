// Tests for the rolling-transcript helpers.
//
// ends_mid_thought() decides whether to extend the end-of-utterance wait, so
// its two failure directions are not symmetric: a false positive costs 400 ms
// of extra latency, a false negative cuts the user off mid-sentence. The cases
// below lean on keeping the second kind rare.
#include "transcript.h"

#include <cstdio>
#include <string>

static int g_fail = 0;

static void eq(const std::string & name, const std::string & got, const std::string & want) {
    const bool ok = got == want;
    if (!ok) g_fail++;
    printf("%-56s %s  got=\"%s\" want=\"%s\"\n", name.c_str(), ok ? "PASS" : "FAIL",
           got.c_str(), want.c_str());
}

static void is(const std::string & name, bool got, bool want) {
    const bool ok = got == want;
    if (!ok) g_fail++;
    printf("%-56s %s  (%d, want %d)\n", name.c_str(), ok ? "PASS" : "FAIL",
           (int) got, (int) want);
}

int main() {
    using namespace transcript;

    // normalise: word-boundary padding is what keeps wake matching honest.
    eq("normalise pads and lowercases",      normalise("Hey Gemma"),      " hey gemma ");
    eq("normalise collapses punctuation",    normalise("hey, gemma!"),    " hey gemma ");
    eq("normalise collapses runs of space",  normalise("hey   gemma"),    " hey gemma ");
    eq("normalise keeps digits",             normalise("gemma 42"),       " gemma 42 ");
    eq("normalise of empty",                 normalise(""),               " ");
    eq("normalise of punctuation only",      normalise("!!!"),            " ");

    // last_word
    eq("last_word of multi-word",            last_word(" the quick fox "), "fox");
    eq("last_word of single word",           last_word(" and "),           "and");
    eq("last_word of empty",                 last_word(" "),               "");

    // The reason this exists: transcripts that clearly continue.
    is("'and' is mid-thought",     ends_mid_thought(normalise("i want to know about paris and")), true);
    is("'the' is mid-thought",     ends_mid_thought(normalise("what is the")),                    true);
    is("'to' is mid-thought",      ends_mid_thought(normalise("i need to")),                      true);
    is("'because' is mid-thought", ends_mid_thought(normalise("that happened because")),          true);
    is("'is' is mid-thought",      ends_mid_thought(normalise("the capital is")),                 true);

    // Complete utterances must NOT extend the wait — that is the costly
    // direction, since every false positive here delays every turn.
    is("complete sentence is not mid-thought",
       ends_mid_thought(normalise("what is the capital of france")), false);
    is("trailing period does not matter",
       ends_mid_thought(normalise("the capital is paris.")),         false);
    is("single content word is not mid-thought",
       ends_mid_thought(normalise("paris")),                          false);
    is("empty transcript is not mid-thought",
       ends_mid_thought(normalise("")),                               false);

    // Garbled tails (moonshine hallucinates constantly) must fall through to
    // the default rather than trigger anything.
    is("garbled last word falls through",
       ends_mid_thought(normalise("hughes sensenses how a bicycle stay")), false);
    is("noise word falls through",
       ends_mid_thought(normalise("nonsense how a bicycle stays uprupt")), false);

    // Case and punctuation must not defeat the match.
    is("uppercase still matches",  ends_mid_thought(normalise("I went TO")),   true);
    is("punctuation still matches",ends_mid_thought(normalise("i went to,")),  true);

    printf("\n%s\n", g_fail == 0 ? "all passed" : "FAILURES PRESENT");
    return g_fail == 0 ? 0 : 1;
}
