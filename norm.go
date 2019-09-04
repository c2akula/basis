package basis

import (
	"math"

	"github.com/c2akula/basis/nd"
)

// Norm computes the Euclidean norm of the array, x.
func Norm(x nd.Array) (s float64) { return math.Sqrt(Dot(x, x)) }
