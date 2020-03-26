# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= 
SPHINXBUILD   ?= sphinx-build
#SPHINXBUILD   ?= python3 -m sphinx
SOURCEDIR     = .
BUILDDIR      = ./web

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

PBE:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" ../../PBE-Documentation/docs $(O)

EE:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" ../../EE-UQ-Documentation/docs $(O)

WE:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" ../../WE-UQ-Documentation/docs $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
