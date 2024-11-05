Welcome to the pelicun developer guide.
The following pages contain information on setting up a development environment, our code quality requirements, our code of conduct, and information on the submission process.
Thank you for your interest in extending pelicun.
We are looking forward to your contributions!

===============
Getting started
===============

Code of conduct
===============

Our pledge
----------

We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone, regardless of age, body size, visible or invisible disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socioeconomic status, nationality, personal appearance, race, caste, color, religion, or sexual identity and orientation.

We pledge to act and interact in ways that contribute to an open, welcoming, diverse, inclusive, and healthy community.

Our standards
-------------

Examples of behavior that contributes to a positive environment for our community include:

-  Demonstrating empathy and kindness toward other people.
-  Being respectful of differing opinions, viewpoints, and experiences.
-  Giving and gracefully accepting constructive feedback.
-  Accepting responsibility and apologizing to those affected by our mistakes, and learning from the experience.
-  Focusing on what is best not just for us as individuals, but for the overall community.

Examples of unacceptable behavior include:

-  The use of sexualized language or imagery, and sexual attention or advances of any kind.
-  Trolling, insulting or derogatory comments, and personal or political attacks.
-  Public or private harassment.
-  Publishing others’ private information, such as a physical or email address, without their explicit permission.
-  Other conduct which could reasonably be considered inappropriate in a professional setting.

Enforcement responsibilities
----------------------------

Community leaders are responsible for clarifying and enforcing our standards of acceptable behavior and will take appropriate and fair corrective action in response to any behavior that they deem inappropriate, threatening, offensive, or harmful.

Community leaders have the right and responsibility to remove, edit, or reject comments, commits, code, wiki edits, issues, and other contributions that are not aligned to this Code of Conduct, and will communicate reasons for moderation decisions when appropriate.

Scope
-----

This Code of Conduct applies within all community spaces, and also applies when an individual is officially representing the community in public spaces.
Examples of representing our community include using an official email address, posting via an official social media account, or acting as an appointed representative at an online or offline event.

Enforcement
-----------

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported to the community leader responsible for enforcement at ``adamzs@stanford.edu``.
All complaints will be reviewed and investigated promptly and fairly.

All community leaders are obligated to respect the privacy and security of the reporter of any incident.

Enforcement guidelines
----------------------

Community leaders will follow these Community Impact Guidelines in determining the consequences for any action they deem in violation of this Code of Conduct:

1. Correction
~~~~~~~~~~~~~

**Community Impact**: Use of inappropriate language or other behavior deemed unprofessional or unwelcome in the community.

**Consequence**: A private, written warning from community leaders, providing clarity around the nature of the violation and an explanation of why the behavior was inappropriate.
A public apology may be requested.

2. Warning
~~~~~~~~~~

**Community Impact**: A violation through a single incident or series of actions.

**Consequence**: A warning with consequences for continued behavior.
No interaction with the people involved, including unsolicited interaction with those enforcing the Code of Conduct, for a specified period of time.
This includes avoiding interactions in community spaces as well as external channels like social media.
Violating these terms may lead to a temporary or permanent ban.

3. Temporary ban
~~~~~~~~~~~~~~~~

**Community Impact**: A serious violation of community standards, including sustained inappropriate behavior.

**Consequence**: A temporary ban from any sort of interaction or public communication with the community for a specified period of time.
No public or private interaction with the people involved, including unsolicited interaction with those enforcing the Code of Conduct, is allowed during this period.
Violating these terms may lead to a permanent ban.

4. Permanent Ban
~~~~~~~~~~~~~~~~

**Community Impact**: Demonstrating a pattern of violation of community standards, including sustained inappropriate behavior, harassment of an individual, or aggression toward or disparagement of classes of individuals.

**Consequence**: A permanent ban from any sort of public interaction within the community.

Attribution
-----------

This Code of Conduct is adapted from the `Contributor Covenant <https://www.contributor-covenant.org>`__, version 2.1, available at https://www.contributor-covenant.org/version/2/1/code_of_conduct.html.

Community Impact Guidelines were inspired by `Mozilla’s code of conduct enforcement ladder <https://github.com/mozilla/diversity>`__.

For answers to common questions about this code of conduct, see the FAQ at https://www.contributor-covenant.org/faq.
Translations are available at https://www.contributor-covenant.org/translations.

.. _contributing:

How to contribute
=================

Prerequisites
-------------

Contributing to pelicun requires being familiar with the following:

.. dropdown:: Python Programming

   Being familiar with object-oriented programming in Python, the PDB debugger, and having familiarity with Numpy and Pandas to handle arrays and DataFrames.

   The following resources may be helpful:

   - `python.org tutorial <https://docs.python.org/3/tutorial/index.html>`_
   - `numpy beginner's guide <https://numpy.org/doc/stable/user/absolute_beginners.html>`_
   - `numpy user guide <https://numpy.org/doc/stable/user/index.html#user>`_
   - `pandas beginner's guide <https://pandas.pydata.org/docs/getting_started/index.html#getting-started>`_
   - `pandas user guide <https://pandas.pydata.org/docs/user_guide/index.html#user-guide>`_

.. dropdown:: Virtual Environments

   Managing a development environment, installing and removing packages.

   The following resources may be helpful:

   - `Python: Virtual Environments and Packages <https://docs.python.org/3/tutorial/venv.html>`_

   - `Conda: Managing environments <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
   - `Micromamba User Guide <https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html>`_

.. dropdown:: reStructured Text Markup

   Being able to extend the documentation by reviewing the existing files and following the same pattern, without introducing compilation warnings.

   The following resources may be helpful:

   - `reStructuredText documentation <https://docutils.sourceforge.io/rst.html>`_
   - `Sphinx User Guide: Using Sphinx <https://www.sphinx-doc.org/en/master/usage/index.html>`_

.. dropdown:: Git for Version Control

   Knowing how to clone a repository, create and checkout branches, review commit logs, commit well-documented changes, or stashing them for later use.
   The following may be useful:

   `git reference manual <https://git-scm.com/docs>`_

.. dropdown:: Command Line

   Being able to set up and call command-line programs beyond Git, including the linting and formatting tools we use.

   The following resources may be useful:

   .. tab-set::

      .. tab-item:: Linux & Mac

         `Bash Reference Manual <https://www.gnu.org/software/bash/manual/html_node/index.html>`_

      .. tab-item:: Windows

         `PowerShell Documentation <https://learn.microsoft.com/en-us/powershell/>`_

.. dropdown:: Pattern-matching

   Ultimately, we learn by example.
   The files already present in the pelicun source code offer an existing template that can help you understand what any potential additions should look like.
   Actively exploring the existing files, tinkering them and breaking things is a great way to gain a deeper understanding of the package.

Contributing workflow
---------------------

The development of pelicun is done via Pull Requests (PR) on GitHub.
Contributors need to carry out the following steps to submit a successful PR:

- `Create a GitHub account <https://github.com/signup>`_, if you don't already have one.
- `Fork the primary pelicun repository <https://github.com/NHERI-SimCenter/pelicun/fork>`_.
- On the fork, create a feature branch with an appropriate starting point.
- Make and commit changes to the branch.
- Push to your remote repository.
- Open a well-documented PR on the GitHub website.

If you are working on multiple features, please use multiple dedicated feature branches with the same starting point instead of lumping them into a single branch.
This approach substantially simplifies the review process, and changes on multiple fronts can be easily merged after being reviewed.
On each feature branch, please commit changes often and include meaningful commit titles and messages.

.. tip::

   Consider taking advantage of advanced Git clients, which enable selective, partial staging of hunks, helping organize commits.

   .. dropdown:: Potential options

      - `Emacs Magit <https://magit.vc/manual/magit.html>`_, for Emacs users. Tried and true, used by our development team.
      - `Sublime Merge <https://www.sublimemerge.com/>`_, also used by our development team.
      - `GitHub Desktop <https://github.com/apps/desktop>`_, convenient and user-friendly.

Code review process
-------------------

After you submit your PR, we are going to promptly review your commits, offer feedback and request changes.
All contributions code need to be comprehensive.
That is, inclusion of new objects, methods, or functions should be accompanied by unit tests having reasonable coverage, and extension of the documentation pages as appropriate.
We will direct you to extend your changes to cover those areas if you haven't done so.
After the review process, the PR will either be merged to the main repository or rejected with sufficient justification.

We will accept any contribution that we believe ultimately improves pelicun, no matter how big or small.
You are welcome to open a PR even for a single typo.

Identifying contribution opportunities
--------------------------------------

The `Issues <https://github.com/NHERI-SimCenter/pelicun/issues>`_ page on GitHub documents areas needing improvement.
If you are interested in becoming a contributor but don't have a specific change in mind, feel free to work on addressing any of the issues listed.
If you would like to offer a contribution that extends the fundamental framework, please begin by `initiating a discussion <https://github.com/orgs/NHERI-SimCenter/discussions/new?category=pelicun>`_ before you work on changes to avoid making unnecessary effort.
