# TODO

+ General enhancements:
    * Adapt to Eigen 5.0
    * Simplify the CMakeLists.txt
    * Update the README
    * Write documentation
    * Create deb/rpm packages

+ Missing features:
    * Add support for normal-tangential coordinates
    * Add non-dimensionalized formulation

+ Not urgent:
    * Remove boost dependency to reduce complexity of installation?
        1. Replace boost::python with pybind11
        2. Remove eigenpy library
        3. Find alternative to boost::format
