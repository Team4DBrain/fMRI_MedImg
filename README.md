# CAI-MedImg

CAI-MedImg is a modular medical-imaging project for testing different fMRI
processing pipelines and comparing which approach gives better results.

Each team member develops one pipeline component in a separate subfolder. The
components should expose a simple interface so they can be combined and tested
against each other in the full project.

Current modules:

- `data_interpolation/`: temporal fMRI interpolation. It takes a 4D BOLD NIfTI
  file and generates a new file with interpolated time frames.


The main goal is to run different pipeline choices, evaluate their outputs, and
keep the best-performing methods for the final system.
