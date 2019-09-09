package nd

import (
	"fmt"
	"testing"
)

type iterator struct {
	x      *Ndarray  // reference to array
	istr   Shape     // iterator strides
	stri   []float64 // inverse iterator strides
	ds, dn Shape     // strides and lengths of the dimensions
	dep    int       // depth of the iterator
	size   int       // # of iterations
	k      int       // track iterations
	sub    Index     // track the start of the iterating dimension
}

func newiterator(x *Ndarray, depth ...int) iterator {
	d := 1
	if len(depth) > 0 {
		if _d := depth[0]; _d > 0 && _d < x.ndims {
			d = _d
		} else {
			panic("invalid iterator depth specified. must be a value between [1,ndims-1], inclusive")
		}
	}
	ndit := x.ndims - d
	it := iterator{
		x:    x,
		istr: ComputeStrides(x.shape[:ndit]),
		stri: make([]float64, ndit),
		dep:  d,
		size: ComputeSize(x.shape[:ndit]),
		sub:  make(Index, ndit),
		ds:   x.strides[ndit:],
		dn:   x.shape[ndit:],
	}
	for i, n := range it.istr {
		it.stri[i] = 1 / float64(n)
	}
	return it
}

func (it *iterator) init() (v []float64, ok bool) {
	it.k = 0
	return it.x.data, true
}

func (it *iterator) next() (v []float64, ok bool) {
	it.k++
	if it.k < it.size {
		return it.x.data[it.ind(it.k):], true
	}
	return nil, false
}

func TestIterator(t *testing.T) {
	shp := Shape{3, 4, 5}
	x := Arange(0, float64(ComputeSize(shp))).Reshape(shp)
	fmt.Println("x: ", x)
	it := newiterator(x, 2)

	for v, ok := it.init(); ok; v, ok = it.next() {
		for i := 0; i < it.dn[0]; i++ {
			k := i * it.ds[0]
			for j := 0; j < it.dn[1]; j++ {
				fmt.Println(v[k+j*it.ds[1]])
			}
		}
	}
}

func (it *iterator) ind(k int) (s int) {
	for i, n := range it.stri {
		j := int(float64(k) * n)
		s += j * it.x.strides[i]
		k -= j * it.istr[i]
		it.sub[i] = j
	}
	return
}

func BenchmarkIterator(b *testing.B) {

	b.Run("depth = 2", func(b *testing.B) {
		b.ReportAllocs()
		x := Rand(TestArrayShape)
		s := 0.0
		it := newiterator(x, 2)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			s = 0
			for xv, ok := it.init(); ok; xv, ok = it.next() {
				for k := 0; k < it.dn[0]; k++ {
					l := k * it.ds[0]
					for j := 0; j < it.dn[1]; j++ {
						v := xv[l+j*it.ds[1]]
						s += v * v
					}
				}
			}
		}
		_ = s * s
	})

	b.Run("depth = 1", func(b *testing.B) {
		b.ReportAllocs()
		x := Rand(TestArrayShape)
		s := 0.0
		it := newiterator(x)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			s = 0
			for xv, ok := it.init(); ok; xv, ok = it.next() {
				for j, n := 0, it.dn[0]; n != 0; j += it.ds[0] {
					v := xv[j]
					s += v * v
					n--
				}
			}
		}
		_ = s * s
	})
}

func TestZipIter(t *testing.T) {
	shp := Shape{3, 4, 5}
	x := Arange(0, float64(ComputeSize(shp))).Reshape(shp)
	y := Arange(0, float64(ComputeSize(shp))).Reshape(shp)
	fmt.Println("x: ", x, "y: ", y)
	it := Zip(x, y)
	for inc := it.Stride(); !it.Done(); {
		xv, yv, n := it.Next()
		for j := 0; n != 0; j += inc {
			fmt.Printf("xv: %v    yv: %v\n", xv[j], yv[j])
			n--
		}
	}
}

func TestNditer(t *testing.T) {
	shp := Shape{3, 4, 5}
	x := Arange(0, float64(ComputeSize(shp))).Reshape(shp)
	fmt.Println("x: ", x)
	for it := NewNditer(x); it.Next(); {
		v, n := it.Get()
		for j := 0; n != 0; j += it.Stride() {
			fmt.Println(v[j])
			n--
		}
	}
}

func BenchmarkNditer(b *testing.B) {
	b.ReportAllocs()
	x := Rand(TestArrayShape)
	it := NewNditer(x)
	inc := it.Stride()
	s := 0.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for it.Reset(); it.Next(); {
			xv, n := it.Get()
			for j := 0; n != 0; j += inc {
				v := xv[j]
				s += v * v
				n--
			}
		}
	}
	_ = s * s
}
