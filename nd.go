package basis

// _nel specifies a limit on the no. of elements, below which
// we'll use the Array's Iterator object to iterate over the
// data to compute the results in the algorithms.
// Beyond this limit, the algorithms are essentially, memory
// bandwidth and cpu cache limited. In other words, straight
// forward, Iterabor based algorithms become less efficient
// and will require some way of processing data in chunks of smaller sizes.
// These algorithms are non-allocating within the routine, unlike
// the Iterator based ones, which allocate memory to compute
// the indices of the elements in the Array that they are referencing.
const _nel = 1e5
