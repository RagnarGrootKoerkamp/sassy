"""Type stubs for sassy Python bindings."""

from typing import Literal

__all__ = ["features", "AllAlignmentsAtPosIter", "Match", "Searcher"]

def features() -> None:
    """Print CPU features and throughput information."""
    ...

class AllAlignmentsAtPosIter:
    """Lazy iterator over all alignments at a single matched end position.

    Do not instantiate directly; instances are obtained from
    :meth:`Searcher.search_all_alignments`.

    All yielded :class:`Match` objects share the same ``text_end``; they differ in
    ``text_start``, ``cigar``, and (when ``margin > 0``) ``cost``.

    Break out of the loop early to avoid enumerating exponentially many paths.
    """

    def __iter__(self) -> "AllAlignmentsAtPosIter": ...
    def __next__(self) -> Match: ...
    def optimal_cost(self) -> int:
        """Return the minimum alignment cost at this end position."""
        ...


class Match:
    """A match result from a search operation."""

    @property
    def pattern_idx(self) -> int:
        """Zero-based index of the pattern that matched."""
        ...

    @property
    def text_idx(self) -> int:
        """Zero-based index of the text where the match was found."""
        ...

    @property
    def text_start(self) -> int:
        """Zero-based inclusive start position of the match in the forward text."""
        ...

    @property
    def text_end(self) -> int:
        """Zero-based exclusive end position of the match in the forward text."""
        ...

    @property
    def pattern_start(self) -> int:
        """Zero-based inclusive start position of the match in the pattern."""
        ...

    @property
    def pattern_end(self) -> int:
        """Zero-based exclusive end position of the match in the pattern."""
        ...

    @property
    def cost(self) -> int:
        """Edit distance cost of the match."""
        ...

    @property
    def strand(self) -> Literal["+", "-"]:
        """Strand direction: '+' for forward, '-' for reverse complement."""
        ...

    @property
    def cigar(self) -> str:
        """CIGAR string describing the alignment."""
        ...

    def __repr__(self) -> str:
        """Return a string representation of the match."""
        ...

class Searcher:
    """A reusable searcher object for fast sequence search."""

    def __init__(
        self,
        alphabet: Literal["ascii", "dna", "iupac"],
        rc: bool = True,
        alpha: float | None = None,
    ) -> None:
        """
        Create a new Searcher.

        Args:
            alphabet: The alphabet to use for searching. One of 'ascii', 'dna', or 'iupac'.
            rc: Whether to search reverse complements (only applies to 'dna' and 'iupac').
            alpha: Alpha parameter for overhang alignments. None for no overhang.
        """
        ...

    def search(self, pattern: bytes, text: bytes, k: int) -> list[Match]:
        """
        Search for a pattern in a text.

        Args:
            pattern: The pattern to search for.
            text: The text to search in.
            k: Maximum edit distance (number of allowed errors).

        Returns:
            A list of Match objects.
        """
        ...

    def search_many(
        self,
        patterns: list[bytes],
        texts: list[bytes],
        k: int,
        threads: int,
        mode: Literal["single", "batch_patterns", "batch_texts"],
    ) -> list[Match]:
        """
        Search multiple patterns in multiple texts using multiple threads.

        Args:
            patterns: List of patterns to search for.
            texts: List of texts to search in.
            k: Maximum edit distance (number of allowed errors).
            threads: Number of threads to use.
            mode: Search mode - 'single', 'batch_patterns', or 'batch_texts'.

        Returns:
            A list of Match objects.
        """
        ...

    def search_all(self, pattern: bytes, text: bytes, k: int) -> list[Match]:
        """
        Search for a pattern in a text, returning all end positions with score <= k.

        This may generate many many matches.
        Only use this instead of `search` if you know what you are doing,
        which typically means there is some postprocessing step to filter overlapping matches.

        Args:
            pattern: The pattern to search for.
            text: The text to search in.
            k: Maximum edit distance (number of allowed errors).

        Returns:
            A list of Match objects.
        """
        ...

    def search_all_alignments(
        self,
        pattern: bytes,
        text: bytes,
        k: int,
        margin: int = 0,
    ) -> list[AllAlignmentsAtPosIter]:
        """
        Search for all end positions with score <= k, returning a lazy iterator
        of all alignments for each end position.

        Returns a list with one :class:`AllAlignmentsAtPosIter` per matched end position.
        Each inner iterator yields :class:`Match` objects for all distinct alignments
        (different CIGAR strings / ``text_start`` values) consistent with that end
        position and cost.

        Iterate lazily and break early to avoid enumerating exponentially many results.

        Args:
            pattern: The pattern to search for.
            text: The text to search in.
            k: Maximum edit distance (number of allowed errors).
            margin: Also yield alignments with cost <= optimal_cost + margin.
                    Clamped to k. Default 0 (optimal alignments only).

        Returns:
            A list of AllAlignmentsAtPosIter objects (one per matched end position).
        """
        ...
