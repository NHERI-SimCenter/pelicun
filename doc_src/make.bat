@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "" goto help

if "%1" == "RDT" (
	%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
	robocopy %BUILDDIR%/html ../../RDT-Documentation/docs /E > nul
	echo.Generating RDT Documentation...
	goto end
)

if "%1" == "PBE" (
	%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
	robocopy %BUILDDIR%/html ../../PBE-Documentation/docs /E > nul
	echo.Generating PBE Documentation...
	goto end
)

if "%1" == "quoFEM" (
	%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
	robocopy %BUILDDIR%/html ../../quoFEM-Documentation/docs /E > nul
	echo.Generating quoFEM Documentation...
	goto end
)

if "%1" == "EE" (
	%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
	robocopy %BUILDDIR%/html ../../EE-UQ-Documentation/docs /E > nul
	echo.Generating EE-UQ Documentation...
	goto end
)

if "%1" == "WE" (
	%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
	robocopy %BUILDDIR%/html ../../WE-UQ-Documentation/docs /E > nul
	echo.Generating WE-UQ Documentation...
	goto end
)

if "%1" == "pelicun" (
	%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
	robocopy %BUILDDIR%/html ../docs /E > nul
	echo.Generating pelicun Documentation...
	goto end
)

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
