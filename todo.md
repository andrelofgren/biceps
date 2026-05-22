# TODO

+ General enhancements:
    * Simplify the CMakeLists.txt
    * Update the README
    * Rewrite code documentation*
        1. Remove all comments
        2. Write new comments
        3. Replace doxygen with sphinx

*Much of the code documentation is currently AI slop and some comments quite misleading, and therefore I aim to replace it with my own human slop at some point.

+ Missing features:
    * Add support for normal-tangential coordinates
    * Add non-dimensionalized formulation

+ Not urgent:
    * Remove boost dependency to reduce complexity of installation?
        1. Replace boost::python with pybind11
        2. Remove eigenpy library
        3. Find alternative to boost::format
