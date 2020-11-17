.. _lblVerification:

************
Verification
************

The methods in pelicun are verified with Unit and Integration tests after each commit using the Travis-CI Cloud-based Continuous Integration platform. The tests performed are grouped by modules and listed under *Continuous Integration* below. These verification tests confirm that the methods perform the tasks as we intended. Additional tests will be added under *Benchmark problems* that compare the outputs of pelicun to other, established performance assessment tools, such as PACT and SP3. These benchmark problems will show that the results are in line with those you would expect from the other tools available.

Continuous Integration
----------------------

Each of the pages below lists the tests performed in pelicun and provides a short description of the tested functionality. The heavily commented test scripts and all required input data are available in the test folder of the pelicun package.

.. toctree::
	:maxdepth: 1

	control <test_control>
	db <test_db>
	file_io <test_file_io>
	model <test_model>
	uq <test_uq>

Benchmark problems
------------------

.. toctree::
	:maxdepth: 1

	SF Tall Building Study <benchmark_SFTB>