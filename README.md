Bumping
=======
This project requires the Weka libraries.  Weka.jar (available online) should be included in Referenced Libraries.

This project implements bumping (see http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.35.8633&rep=rep1&type=pdf) for J48 trees and consistency subset analysis.

The project has two different files which are run separately.  Main.java implements bumping for J48 trees, and FeatureSelection.java implements bumping for consistency subset evaluation.

The output of bumping with the J48 trees is a .dot file which can be converted to a .png image of the resulting tree.

The output of bumping with CSA is a text file with the names of the features selected.

Currently, the best J48 tree is selected by choosing the smallest tree, but there is code (currently commented out) for choosing the best tree by best fit to the training data.
