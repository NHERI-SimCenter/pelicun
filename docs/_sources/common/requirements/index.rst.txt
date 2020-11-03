Requirements Traceability Matrix
================================


The Requirements Traceability Matrix (RTM) is presented as tables linking requirements with project deliverables.  The requirements for the SimCenter have been obtained from a number of sources:

#. GC: Grand challenges in hazard engineering are the problems, barriers, and bottlenecks that hinder the ideal of a nation resilient from the effects of natural hazards. The vision documents referenced in the solicitation [2, 3, 5, 6] outline the grand challenges for wind and earthquake hazards. These documents all present a list of research and educational advances needed that can contribute knowledge and innovation to overcome the grand challenges. The advances summarized in the vision documents were identified through specially formed committees and workshops comprising researchers and practicing engineers. They identified both the grand challenges faced and also identified what was needed to address these challenges. The software needs identified in these reports that are applicable to research in natural hazards as permitted under the NSF NHERI program were identified in these reports. Those tasks that the NHERI SimCenter identified as pertaining to aiding NHERI researchers perform their research and those which would aid practicing engineers utilize this research in their work are identified here.
#. SP: From the senior personnel on the SimCenter project. The vision documents outline general needs without going into the specifics. From these general needs the senior personnel on the project  identified specific requirements  that would provided a foundation to allow research.
#. UF: SimCenter workshops, boot camps and direct user feedback. As the SimCenter develops and releases tools, feedback from researchers using these tools is obtained at the tool training workshops, programmer boot-camps,  in one-on-one discussions, via direct email, and through online user feedback surveys. 


The software requirements are many. For ease of presentation they are broken into three groups:


\item Regional Scale - Activities to allow researchers to examine the resilience of a community to natural hazard events.
\item Building Scale - Activities to allow researchers to improve on methods related to response assessment and performance based design of individual buildings subject to the impact of a natural hazard.
\item Education - software development activities related to education of researchers and practicing engineers.
\end{enumerate}

\section{Regional Scale Applications}
\input{requirements/bigRequirements.tex}

\clearpage
\section{Building Scale Applications}

For building scale simulations, the requirements are broken down by SimCenter application. There are a number of applications under development for each of the hazards. Many of the requirements related to UQ and nonlinear analysis are repeated amongst the different applications under the assumption that if they are beneficial to engineers dealing with one hazard, they will be beneficial to engineers dealing with other hazards.

\subsection{Response of Building to Wind Hazard}
The following are the requirements for response of single structure due to wind action. The requirements are being met by the WE-UQ application. All requirements in this section are related to work in WBS 1.3.7.

\input{requirements/WEUQ_Requirements.tex}
 
\clearpage
\subsection{Response of Building to Hydrodynamic Effects Due to Tsunami or Coastal Inundation}
The following are the requirements for response of single structure due to hydrodynamic effects of water caused earthquake induced tsunami or coastal inundation due to a Hurricane. The requirements are being met by the Hydro-UQ application. All requirements in this section are related to work in WBS 1.3.7.

\input{requirements/HydroUQ_Requirements.tex}
 
\clearpage
\subsection{Response of Building to Earthquake Hazard}
The following are the requirements for response of single structure to earthquake hazards. The requirements are being met by the EE-UQ application. All requirements in this section are related to work in WBS 1.3.8.

\input{requirements/EEUQ_Requirements.tex}

\clearpage
\subsection{quoFEM}
The following are the requirements are being for the quoFEM application. quoFEM is an application proving UQ and Optimization methods to existing FEM applications. uqFEM has a lower level interface to UQ and Optimization methods than the other applications (WE-UQ, EE-UQ, and PBE). It is thus a more powerful tool providing more capabilities for researchers. All requirements in this section are related to work in WBS 1.3.8.

 \input{requirements/QUO_FEM_Requirements.tex}


\clearpage
\subsection{Performance Based Engineering}
The following are the requirements for application(s) related to performance based engineering of a single structure related to natural hazards such as earthquake and hurricane . The requirements are being met by the PBE application. All requirements in this section are related to work in WBS 1.3.9.

\input{requirements/PBE_Requirements.tex} 

\section{Educational Software}

The following are educational activities obtained that are related to software development.

\input{requirements/edRequirements.tex} 

Contact
=======
Frank McKenna, NHERI SimCenter, UC Berkeley, fmckenna@berkeley.edu