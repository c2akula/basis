package nd

import (
	"fmt"
	"io"
)

type (
	Shape []int
	Index []int

	Array interface {
		// Metadata
		// Data provides access to the internal data buffer.
		Data() []float64
		// Shape returns a slice containing the sizes of each dimension.
		Shape() Shape
		// Strides returns a slice containing the step sizes along each dimension.
		Strides() Shape
		// Size returns the no. of elements in the Array.
		Size() int
		// Ndims returns the dimensionality of the Array.
		Ndims() int

		// Element and View Access
		// At returns a pointer to the element at the specified cartesian index.
		At(index Index) *float64
		// Get returns the element at the specified cartesian index.
		Get(index Index) float64
		// Set writes the value f at the specified cartesian index.
		Set(f float64, index Index)
		// View extracts a sub-Array of the specified shape, starting at the cartesian Index, start.
		View(start Index, shape Shape) *Ndarray

		// Iterator
		Iterable

		// String
		fmt.Stringer

		// Read/Write
		io.ReadWriter
	}

	View interface {
		Array
	}

	Ndarray struct {
		data    []float64 // shape * dsize
		size    int       // # of elements
		ndims   int       // # of dimensions
		strides Shape     // strides * dsize
		shape   Shape     // dimension sizes
	}

	Iterable interface {
		Iter() *Iter
	}
)
