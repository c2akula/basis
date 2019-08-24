package nd

import (
	"fmt"
	"io"
)

type (
	Shape []int
	Index []int

	Array interface {
		Type

		// Metadata
		Data() []float64
		Shape() Shape
		Strides() Shape
		Ndims() int

		// Element and View Access
		Get(Index) float64
		Set(float64, Index)
		View(start Index, shape Shape) Array

		// Iterator
		Iterable
		Begin() Index
		End() Index
	}

	Ndarray struct {
		size     int       // # of elements
		data     []float64 // shape * dsize
		ndims    int       // # of dimensions
		strides  Shape     // strides * dsize
		shape    Shape     // dimension sizes
		it       Iterator
		beg, end Index
	}

	// A Type is any object that has a size and can be represented through bytes.
	Type interface {
		// Size returns the no. of bytes used to represent the underlying type.
		Size() int
		io.ReadWriter
		fmt.Stringer
	}
	Iterable interface {
		Take() Iterator
	}
)
