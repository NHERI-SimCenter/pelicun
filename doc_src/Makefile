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

RDT:
	make clean
	rm -fr ../../RDT/docs/_sources
	rm -fr ../../RDT/docs/_static
	rm -fr ../../RDT/docs/_images
	rm -fr ../../RDT/docs/common
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" ../../PBE-Documentation/docs $(O)

PBE:
	make clean
	rm -fr ../.../PBE-Documentation/docs/_sources
	rm -fr ../../PBE-Documentation/docs/_static
	rm -fr ../../PBE-Documentation/docs/_images
	rm -fr ../../PBE-Documentation/docs/common
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" ../../PBE-Documentation/docs $(O)

quoFEM:
	make clean
	rm -fr ../.../quoFEM-Documentation/docs/_sources
	rm -fr ../../quoFEM-Documentation/docs/_static
	rm -fr ../../quoFEM-Documentation/docs/_images
	rm -fr ../../quoFEM-Documentation/docs/common
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" ../../quoFEM-Documentation/docs $(O)

EE:
	make clean
	rm -fr ../../EE-UQ-Documentation/docs/_sources
	rm -fr ../../EE-UQ-Documentation/docs/_static
	rm -fr ../../EE-UQ-Documentation/docs/_images
	rm -fr ../../EE-UQ-Documentation/docs/common
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" ../../EE-UQ-Documentation/docs $(O)

WE:
	make clean
	rm -fr ../../WE-UQ-Documentation/docs/_sources
	rm -fr ../../WE-UQ-Documentation/docs/_static
	rm -fr ../../WE-UQ-Documentation/docs/_images
	rm -fr ../../WE-UQ-Documentation/docs/common
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" ../../WE-UQ-Documentation/docs $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
