
.. _lblArchitecture:

*********************
Software Architecture
*********************

The SimCenter is developing a software framework for building scientific workflow applications to perform computational; simulations in field of NHE at both building level scale and regionsl scale. It is releasing a number of applications built using this framework. The |app| is one of those applications which have been released (EE-UQ, WE-UQ, PBE). Other applications are under development (RDT). The applications that the SimCenter are developing are limited scientific workflow systems. This chapter presents the software architecture for the framework and the |app| built using it using the C4 model.

.. note:: 

   1.   **Workflow**: A sequence of steps involved in moving from a beginning state to an ending state.

   2. **Scientific Workflow Application**: An application that automates a workflow process through software, with each step in the workflow being performed by a separate “scientific” software application.

   3. **Scientific Workflow System**: software providing an infrastructure for the set-up, scheduling, running, and monitoring of a user defined scientific workflow application.

Level 1: A Context for SimCenter Applications
=============================================

A Level 1 diagram showing the system context for the SimCenter applications, i.e. how it fits in the world,  is shown in :numref:`figContext`. It shows SimCenter applications (EE-UQ, WE-UQ, PBE, RDT) as a box in the center surrounded by the user and the systems it and the user interact with. The SimCenter applications allows user to create and run scientific workflow applications, the data for the applications may be obtained from the web or DataDepot, the workflow applications are run on either the local desktop or on some HPC at |DesignSafe|. 

.. _figContext:

.. figure:: figures/context.png
   :align: center
   :figclass: align-center

   System Context Diagram for SimCenter Applications

Level 2:  The Components of a SimCenter Application
===================================================

Given how SimCenter applications fit in with the environment, a level 2 diagrams now demonstrates how the SimCenter applications are broken into high level components. The SimCenter applications are, as shown in :numref:`figContainer`, broken into two components: A front end UI and a back end application that runs the workflow. The front end applications are desktop applications written using the cross-platform Qt framework. The back end is an application that processes the input from the front end, which comes in the form of a JSON file, creates a workflow and runs it. The workflow applications, written in Python, C, or C++, utilize existing applications were possible and run on either the local desktop machine or on a HPC utilizing resources made available to NHE community through DeisignSafe. 

.. _figContainer:

.. figure:: figures/container.png
   :align: center
   :figclass: align-center

   System Container Diagram for SimCenter Applications

Level 3: Container Diagrams for the Front and Back End Components
=================================================================

Two level 3 diagrams are now presented which break up the two containers into the major building blocks or components in C4 terminology. In :numref:`figComponentFront` the component diagram for the front end UI is presented. It outlines the interaction between the user and the individual graphical elements (widgets) of the UI. Given the analogy of a jigsaw puzzle, the user selects which piece of the jigsaw puzzle they are working on in the component selection widget. The widget for the jigsaw piece will then be displayed on the desktop. The user for each jigsaw piece then selects which application to run for that piece, and for the chosen application, they provide the inputs. When the inputs are all provided, the user can select to run the simulations locally or remotely. For jobs that run remotely, the user can download and review previously run simulations. As seen the widgets may subsequentially interact with web services through HTTPS requests, or with DesignSafe utilizing TAPIS Restful API through the RemoteService container.

.. _figComponentFront:

.. figure:: figures/componentFront.png
   :align: center
   :figclass: align-center

   Component Diagram for Front End UI

The component diagram for the backend application shown in :numref:`figComponentBack`, shows that the backend is made up of a number of component, all applications. The application “femUQ.py” is the application that parses the input from the front end, sets up the workflow and launches the UQ engine. Which UQ Engine and which applications to run in the workflow, is determined from the data passed from the UI and information contained in a WorkflowApplication.json file. A file is used to allow the researchers to modify the applications that may be run in the workflow w/o the need to recompile the application. Control is then passed to a UQ Engine, which repeatedly runs the workflow to generate the results. In running the workflow some of the applications will invoke applications not developed to meet the API. For such applications pre- and post-processors are provided.
The figure shows the backend application running locally or remotely on a HPC@DesignSafe.

 
.. _figComponentBack:

.. figure:: figures/componentBack.png
   :align: center
   :figclass: align-center

   Component Diagram for Backend Application

Level 4 UML Diagrams
====================

A number of diagrams are presented for the level 4 diagrams. These are mostly UML diagrams showing how the applications are built. The SimCenter releases a number of front-end applications: EE-UQ shown in \:numref:`figUmlEE`, WE-UQ shown in :numref:`figUmlWE`, and PBE shown in :numref:`figUmlPBE`. These applications share code with each other and other SimCenter applications. As a consequence, the common code is bundled into a number of shared packages: EarthquakeEvents shown in :numref:`figUmlEarthquakeEvents`, WindEvents shown in :numref:`figUmlWindEvents`, and SimCenterCommon shown in :numref:`figUmlCommon`. A number of packages were chosen over placing all common code inside a single package to simplify development efforts for outside programmers (whom it is envisioned will mostly be adding new event components) and to reduce the overhead of package management and compile time for SimCenter programmers. UML diagrams are  presented for these front-end applications and shared packages. THE UML diagrams that are presented are not exhaustive, in that they do not show all classes used, for it was decided not to for example show the myriad of Line edits, labels, spin boxes, etc. that make up the widgets. What is shown is sufficient to present the SimCenter architecture.

While there are a number of different types of UML diagrams,  those shown in this document will be limited to class diagrams and sequence diagrams. SimCenter applications are object-oriented in nature. An object-oriented program consists of objects interacting with one another,  with each object being of a certain type or class. A class diagram shows the classes, their attributes and methods, and the relationships between the classes. A sequence diagram or event diagram shows the order in which objects interact. To understand the SimCenter framework it is useful to first present the main() function for a SImCenter application, in this case EE-UQ, shown in :numref:`codeMainCode`. The code presentebd is a stripped down version of the actual code, code for dealing with style sheets, analytics, etc. is not shown as it is not pertinent to understanding of the software architecture.


.. _codeMainCode:

.. code-block::
   
   int main(int argc, char *argv[]) {

     QApplication app(argc, argv);
 
    //                                                                       
    // create a remote interface                                             
    //                                                                       

    QString tenant("designsafe");
    QString storage("agave://designsafe.storage.default/");
    QString dirName("EE-UQ");
    
    //                                                                       
    // create the main window                                                
    // 
    
    WorkflowAppWidget *theInputApp = new WorkflowAppEE_UQ(theRemoteService);
    MainWindowWorkflowApp window(QString("EE-UQ: Response of Building to Earthquake"), theInputApp, theRemoteService);
    
    windows.setVersion("Version 1.0.0");


    //                                                                       
    // move remote interface to a thread                                     
    //                                                                       

    QThread *thread = new QThread();
    theRemoteService->moveToThread(thread); 
    thread->start();

    //                                                                       
    // show the main window, set styles & start the event loop               
    //                                                                       

    window.show(); 
    int res = app.exec();

    //                                                                       
    // on done with event loop, logout & stop the thread                     
    //                                                                       

    theRemoteService->logout();
    thread->quit();
    
     return res;
   }


As was mentioned the Front end UI applications are built using Qt. In a Qt application the programmer creates a QApplication object, in :numref:`codeMainCode` the object named `app` and a QMainWindow, in the example named `window`. As will be shown in :numRef:`figUmlCommon`, MainWindowWorkflowApp is a type of QMainWindow that is used in all SimCenter research applications as it deals with all the application menu items, e.g. File open and close, Help cite, etc The QMainWindowWorkflowApp is a SImCenter class that contains a single QWidget of type WorkflowAppWidget. The WorkflowAppWidget object is passed a RemoteService, the remote cloud service that the application will interact with. This RemoteService is placed in it's own QThread object, so that the UI can respond to user requests while communication with cloud service is underway. Once the window object is shown, control is passed to the QApplication  until the user is done.

For |app| the type of WorkflowAppWidget is of type |workflowWidgetAPP|, which is shown in :numref:`figUmlEE`. Other applications have their own subclasses of WorkflowAappWidget.


.. _lblUmlEE:


UML EE-UQ
---------

EE-UQ is an application to determine the response of a building subjected to an earthquake event. As shown in :numref:`figumlEE` it comprises a component selection which presents the user with a a number of components, jigsaw pieces, which include: earthquake event (EarthquakeEventSelection), UQ engine (UQ Selection), demand parameters of inters (EDP Selection), building information model (BIM Selection),  strutctural analysis model generator (SAM Selection), finite element application (FEM Selection), and RandomVariableContainer.  RandomVariableContainer is a widget allowing user to specify distrubutions associated with the random variables created by user. As will be seen in :numref:`figUmlEarthquakeEvents` and :numref:`figUmlCommon` each component offers the user a number of applications to choose from for that component. Other classes corresponding to widgets presented in the Front end UI include: UQ Result for displaying the results, Local and Remote Services for running the job locally or remotely, Remote job Manager for monitoring job status and retrieving old jobs, and Login for obtaining credentials from DesignSafe to access and run jobs on the HPC resources. All communication between the applications and DesignSafe-ci is through the Application Service. This is done to allow the applications to switch to other cloud service providers, possibly allowing applications to run at DesignSafe, on Amazon EC-2, IBM's Azure or elsewhere.

.. _figUmlEE:

.. figure:: figures/umlEE.png
   :align: center
   :figclass: align-center

   UML Diagram for EU-UQ

.. _lblUmlWE:

UML WE-UQ
---------

 Similar in  construction to EE-UQ is WE-UQ, as shown in figure :numref:`figumlWE`.  In fact the only difference is that Wind Event Selection is present in the component selection, instead of Earthquake Events. The wind event applications, as will be shown in :numref:`figWindEvents` include stochastic wind models, wind loading from online services such as Vortex-Winds, applications which take online wind tunnel experimental datasets such as those from Tokyo Polytechnic.


.. _figUmlWE:

.. figure:: figures/umlWE.png
   :align: center
   :figclass: align-center

   UML Diagram for WE-UQ







.. only:: PBE

.. _lblUmlPBE:
   
UML PBE
-------

PBE is a tool for performance based engineering. Given a building and an event it will calculate downtime and loss estimates. As can be sen in :numref:`figumlPBE`,  it adds a LossModelSelection to the component Selections available in EE-UQ. In future it, or another application, will add similar for WE-UQ. The Loss Model applications currently available for selection include a a P58 Loss Model and a HAZUS Loss Model. Depending on selection, deifferent widgets are presented for the user to input the different input arguments needed for the different loss model calculations. Presently the calculations for both loss models are perforrmed by the same python script, CalculateDL.py, in the collection of backend applications.

.. _figUmlPBE:

.. figure:: figures/umlPBE.png
   :align: center
   :figclass: align-center

   UML Diagram for PBE


.. _lblUmlEarthquakeEvents

UML EarthquakeEvents
--------------------

The Earthquake Events package, as shown in :numref:`figumlEarthquakeEvents`, contains an Earthquake Event selector with a number of Earthquake Event selections available. The selections include options that interface with the NGA west server directly and options that will collect inputs for stochastic input models of Vlachos et Al or Dabahi and DerKiuerghian, peer NGA records, site response and our own SimCenterEvent format. Each of these widgets corresponds to one application in the backend, e.g. RockOutcrop corresponds to SiteReponse, and it is this application that will run when the workflow runs.

.. _figUmlEarthquakeEvents:

.. figure:: figures/umlEarthquakeEvents.png
   :align: center
   :figclass: align-center

   UML Diagram for Earthquake Events

.. _lblUmlWindEvents:

UML WindEvents
--------------

Similar to the Earthquake Events package, the wind events package shown in :numref:`figumlWindEvents`, contains a WInd Event Selector with a number of Wnd Event selections available. The selections include options for stochastically generated wind events, events that obtain wind loading from the vortex-winds server, options to obtain forces from eind tunnel events, either from the Tokyo Ploytechnic University database, or a user supplied file.

.. _figUmlWindEvents:

.. figure:: figures/umlWindEvents.png
   :align: center
   :figclass: align-center

   UML Diagram for Wind Events

 
.. _lblSimCenterCommon:


SimCenterCommon
---------------

SimCenter common shown in :numref:`figumlCommon} contains a number of component selctions, BIM selection, EDP Selection, SAM Selection, FEM Selection and UQ Engine Selection. Each contains a number of options. The components and their options are all subclasses of the SImCenterAppWidget class, The SImCenterAppWidget has methods to output and input from a JSON object. SimCenterCommon also contains the RandomVariablesContainer class, each object being a container for a number of RandomVariables. Each RansomVariable having a name and a RandomVariable Distribution associated with it. Types of RandomVariableDistributions include for exmaple Normal, Lognormal, Uniform, Beta, and Gumbel.

 
.. _figUmlCommon:

.. figure:: figures/umlCommon.png
   :align: center
   :figclass: align-center

   UML Diagram for SimCenter Common


.. _lblSimCenterBackendApplications:

SimCenter Backend Applications
------------------------------

The BackendApplications are currently all in a single package. These are the applications that perform the numerical computations when the workflow runs. Some of these applications rely on external applications, websites, and external packages.  The external applications, web services, and libraries are as shown in :numref:`figAppDiagramBackend`.


.. _figAppDiagramBackend:

.. figure:: figures/appDiagramBackend.png
   :align: center
   :figclass: align-center

   Applications for Backend Applications
