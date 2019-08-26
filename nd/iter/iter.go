package iter

import "github.com/c2akula/go.nd/nd"

// Iterator provides the ability to access the elements of and iterate over an nd.Array.
type Iterator interface {
	// Len returns the size of the iterator, or the no. of elements referenced by the iterator.
	Len() int

	// Ind returns the positions of the elements referenced by the iterator.
	Ind() nd.Index

	// Data returns the data buffer referenced by the iterator.
	Data() []float64
}

type iter struct {
	len int       // depth of the iterator
	ind nd.Index  // list of indices into the referenced data buffer
	arr []float64 // reference to the array's data buffer iterating upon
}

func (it *iter) Data() []float64 { return it.arr }

func (it *iter) Ind() nd.Index { return it.ind }

func (it *iter) Len() int { return it.len }

// New returns a new iterator for the specified array.
func New(array nd.Array) (Iterator, []float64, nd.Index) {
	it := &iter{
		arr: array.Data(),
		len: array.Size(),
		ind: make(nd.Index, array.Size()),
	}

	ind := make(nd.Index, array.Ndims())
	for i := range it.ind {
		k := nd.Ind2sub(nd.ComputeStrides(array.Shape()), i, ind)
		it.ind[i] = nd.Sub2ind(array.Strides(), k)
	}

	return it, it.arr, it.ind
}
