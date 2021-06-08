# QExplore: A Tool for Automatic Exploration of Dynamic Web Applications By Using Reinforcement Learning
![](https://github.com/salmansherin/QExplore/blob/ec1e06fb936373fcfcf04d9162d859da4f060b20/img.jpg)
## Summary
QExplore is a dynamic automatic exploration tool for dynamic web applications. It reverse engineers a state-flow model that can be used to automate several web analysis and testing techniques. The tool uses a popular reinforcement learning technique, called Q-learning, to systematically explore a web application. To feed the input fields and explore the states behind the web forms, QExplore uses mocker data generator.

## Features
* QExplore targets the exploration of dynamic web applications.
* It also uses mocker data generator for feeding the input fields during the exploration.
* It allows users to define directives to excess parts of the web application restricted by specific user inputs.
* It allows users to define multiple user inputs for single input field. 
* QExplore automatically reverse engineer's a state-flow graph that can be used for testing activities such as test case generation and execution.
* QExplore can be run on any type of dynamic web applications
* QEplore supports multiple browsers i.e. firefox, chrome and IE.

## Limitations
* Currently, QExplore has no support for performing interactions such as Drag and Drop.
* QExplore has no GUI currently and is a work in progress. 
* The minimun time limit for QExplore to explore any dynamic application to provided reasonable coverage is 10 minutes.
