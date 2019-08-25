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
		Data() []float64
		Shape() Shape
		Strides() Shape
		Size() int
		Ndims() int

		// Element and View Access
		Get(Index) float64
		Set(float64, Index)
		View(start Index, shape Shape) Array

		// Fancy Indexing
		TakeAt(index Index) Array

		// Iterator
		Iterable

		// String
		fmt.Stringer

		// Read/Write
		io.ReadWriter
	}

	ndarray struct {
		data    []float64 // shape * dsize
		size    int       // # of elements
		ndims   int       // # of dimensions
		strides Shape     // strides * dsize
		shape   Shape     // dimension sizes
		it      Iterator
	}

	Iterable interface {
		Take() Iterator
	}
)
