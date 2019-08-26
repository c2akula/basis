package go_nd

import (
	"math"

	"github.com/c2akula/go.nd/nd/iter"
)

// Norm computes the Euclidean norm of the array, x.
func Norm(x iter.Iterator) (s float64) { return math.Sqrt(Dot(x, x)) }
