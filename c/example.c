#include "sassy.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static void print_match(const sassy_Match* m, size_t idx) {
	printf("#%zu  pat[%zu-%zu]  txt[%zu-%zu]  cost=%d  strand=%c\n", idx, m->pattern_start,
	       m->pattern_end, m->text_start, m->text_end, m->cost, m->strand == 0 ? '+' : '-');
}

int main() {
	// Create searcher object: dna alphabet, with reverse complements, without overhang.
	sassy_SearcherType* searcher = sassy_searcher("dna", true, NAN);

	// Search a pattern in a text.
	const char* pattern = "AAGGGGA";
	const char* text    = "CCCCCCCCCAAGGGGACCCCCAAGGCGACCCCCCCCC";
	const size_t k      = 1;

	sassy_Match* matches = NULL;

	size_t n_matches = search(searcher, pattern, strlen(pattern), text, strlen(text), k, &matches);

	printf("Found %zu match(es):\n", n_matches);
	for(size_t i = 0; i < n_matches; i++) print_match(&matches[i], i);

	sassy_matches_free(matches, n_matches);
	sassy_searcher_free(searcher);

	return 0;
}
