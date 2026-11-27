# Reproduction and limits

The fixture replay reads only the authored metadata file and writes its
library, bibliography, export, and plot under the local build directory.
The report records the source path and computed counts. The retained
software release has its own identifier, and the new local render records
no publication event.

The fixture tests a shared-identifier case, a distinct entry, and persistence
across a reload. It does not quantify false matches in a real corpus.
Provider-specific behavior, PDF accessibility and extraction, model
summaries, and exploratory analytics require their own inputs and checks.
