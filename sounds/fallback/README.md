# Sound Fallback Directory

This directory contains fallback audio files used when the original sound files cannot be found.

- `silent.wav` - A 1-second silent audio file used as a fallback when required sound effects are missing.

## Required Sound Files

The application expects the following sound files to be present in the parent `sounds` directory:

- `start.wav` - Played when an exercise session starts
- `end.wav` - Played when a single set completes
- `allend.wav` - Played when all exercise sets are completed

If these files are missing, the application will display a warning and use the silent fallback instead.
