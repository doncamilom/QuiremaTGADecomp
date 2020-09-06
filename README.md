# QuiremaTGADecomp
This program has been envisioned as a way to decompose Thermo Gravimetrical Analysis (TGA) curves into simpler logistic curves representing chemical reactions where mass-changes are involved. All of the analysis is performed with the TensorFlow library for the creation of an interpretable NN model and optimization.

The program's aim is to decompose TGA experimental data into logistic curves corresponding to mass gain/loss events. This is achieved with the program with the use of a series of routines written in python, with heavy use of the TensorFlow library.
This modelling must be done for various heating velocities, and once this has been done they must be compared. The final objective is to obtain a linear plot (Arrhenius plots), for which routines are also written that allow to easily transform the NN models into valuable knowledge for chemists.
There are a series of restrictions that must be met by the model, and these are obtained from other experiments or "chemical intuition".

TensorFlow is used to encode the objective function as well as the cost function (built up so as to encode all "chemical restrictions").
Once this encoding has been made it is fairly easy to take advantage of all of the infrastructure implemented in TF.
In the end, this encoding looks like a traditional neural network with a single layer. 

All encoded parameters have a direct physical interpretation, so that the power of this method lies in the interpretability of the NN instead of its performance as a black-box. 

![gif](LatinXChem/TGA_Model_LatinXChem_GIF.gif)
