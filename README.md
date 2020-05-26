# QuiremaTGADecomp
Decompose Thermo Gravimetrical Analysis (TGA) curves into simpler logistic curves representing mass-change events with a chemical interpretation. All of the analysis is performed with the TensorFlow library for the creation of a NN-analogous model and optimization.

The program's aim is to decompose TGA experimental data into logistic curves corresponding to mass gain/loss events. This is achieved with the program with the use of a series of routines written in python, with heavy use of the TensorFlow library.
This modelling must be done for various heating velocities, and once this has been done they must be compared. The final objective is to obtain a linear plot, which the program also does.
There are a series of restrictions that must be met by the model, and these are obtained from other experiments or "chemical intuition".

TensorFlow is used to encode the objective function as well as the cost function (built up so as to encode all "chemical restrictions").
Once this encoding has been made it is fairly easy to take advantage of all of the infrastructure implemented in TF.
In the end, this encoding looks like a traditional neural network but with a custom layer. All encoded parameters have a direct physical interpretation, what makes this model even more powerful.
