# Structural Code Preservation Fixture

The fenced example below is a single semantic code block and should remain intact in the parent store while it is small enough to fit within the configured maximum token limit.

```python
# This fixture uses common words in comments to create a roughly three-hundred-word code block.
# The reader can follow the entire example from the first comment through the final return value.
# A retrieval pipeline should keep this code block whole rather than cutting through the function body.
# Each line gives the parser a clear boundary without changing the syntax of the enclosing function.
# The chunker should inspect the token count before deciding whether an atomic block can be retained.
# When the block fits, its opening fence, function declaration, comments, statements, and closing fence remain together.
# This makes the parent context useful for code retrieval because the surrounding setup is preserved.
# Chunking should prefer structural boundaries such as headings, paragraphs, and complete fenced blocks.
# A child chunk can refer to this parent by a stable identifier without duplicating the full source document.
# The index can then retrieve the precise child while the generator receives the wider parent context.
# Dense retrieval captures similar ideas, while sparse retrieval finds exact identifiers and keywords.
# Rank fusion merges the two rankings, and the cross encoder scores candidate contexts more carefully.
# The final packer checks the context budget and skips a parent that cannot fit without truncation.
# Source metadata remains attached so evaluation can verify which document supported a response.
# This small example is intended for structural verification and is not an executable production module.
# The goal is not to maximize the number of chunks but to keep meaningful units recognizable.
# An oversized code section can be divided between complete functions when a safe boundary is available.
# When a single function itself exceeds the hard cap, split it at safe line or statement boundaries.
# The system should report that split rather than silently pretending an oversized block stayed intact.
# The test suite checks that a block under the hard cap remains intact after ingestion and chunking.
# This helps catch regressions where a token splitter slices the middle of code fences or definitions.
# Parent chunks carry source, heading, and token metadata for diagnostics and context selection.
# Child chunks carry their parent identifier so retrieval can map a narrow match back to its context.
# These relationships are important when the corpus contains many files and repeated technical terms.
# Stable identifiers also support reproducible benchmarks and traceable evaluation reports.
# The example deliberately keeps the code body simple so its structural boundaries are unambiguous.
# The same chunking approach can be extended to abstract syntax trees for language-aware splitting.
# The parser should never execute this text; ingestion treats all source content as untrusted data.
# A production pipeline should also limit file types, document size, and malformed input handling.
# This closing comment leaves the code example structurally complete for the preservation test.
def preserved_code_block_demo():
    """Return a constant to validate that the complete fenced block is retained."""
    status = "structure-preserved"
    return status
```
