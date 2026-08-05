.. _changes_unreleased:

==========
Unreleased
==========

Added
-----


Changed
-------

**Doc Extra Requires Python 3.10**: The ``doc`` extra now installs
only on Python 3.10 and newer. The documentation toolchain has
dropped Python 3.9, so security fixes in those packages no longer
reach Python 3.9-compatible versions and the lock file had to pin
outdated versions for Python 3.9 that kept triggering security
advisories. Pelicun itself and the ``test`` and ``lint`` extras
continue to support Python 3.9; CI builds the documentation on
Python 3.12.


Removed
-------


Fixed
-----

