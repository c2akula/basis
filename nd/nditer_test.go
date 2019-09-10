package nd

import (
	"fmt"
	"testing"
)

type iterator struct {
	x            *Ndarray  // reference to array
	istr         Shape     // iterator strides
	stri         []float64 // inverse iterator strides
	ds, dn       Shape     // strides and lengths of the dimensions
	dep          int       // depth of the iterator
	size         int       // # of iterations
	k            int       // track iterations
	b            int
	sub          Index // track the start of the iterating dimension
	rs, rn       int
	iscontiguous bool
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
		x:            x,
		istr:         ComputeStrides(x.shape[:ndit]),
		stri:         make([]float64, ndit),
		dep:          d,
		size:         ComputeSize(x.shape[:ndit]),
		sub:          make(Index, ndit),
		ds:           x.strides[ndit:],
		dn:           x.shape[ndit:],
		rs:           x.strides[ndit],
		rn:           x.shape[ndit],
		iscontiguous: (d == 1) && (x.strides[ndit] == 1),
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

func (it *iterator) iter() func() ([]float64, bool) {
	it.k = -1
	return func() ([]float64, bool) {
		it.k++
		return it.x.data[it.ind(it.k):], it.k < it.size
	}
}

func (it *iterator) ind(k int) (s int) {
	// if it.iscontiguous {
	// 	return k * it.x.strides[it.x.ndims-2]
	// 	// d := len(it.stri)
	// 	// for i, n := range it.stri[:d-1] {
	// 	// 	j := int(float64(k) * n)
	// 	// 	s += j * it.x.strides[i]
	// 	// 	k -= j * it.istr[i]
	// 	// 	// it.sub[i] = j
	// 	// }
	// 	// fmt.Println("ind:s: ", s)
	// 	// return
	// }
	for i, n := range it.stri {
		j := int(float64(k) * n)
		s += j * it.x.strides[i]
		k -= j * it.istr[i]
		it.sub[i] = j
	}
	return
}

func TestIterator(t *testing.T) {
	shp := Shape{3, 4, 5, 6}
	x := Arange(0, float64(ComputeSize(shp))).Reshape(shp)
	xp := x.Permute([]int{1, 2, 0, 3})
	fmt.Println("xp: ", xp)
	xpshp := make(Shape, xp.ndims)
	copy(xpshp, xp.shape)

	// xv := x.View(Index{1, 0, 1, 2}, Shape{2, 4, 3, 4})
	// fmt.Println("xv: ", xv)
	// y := xv.Permute([]int{3, 2, 1, 0}) // shp: [3,4,2]
	// fmt.Println("y: ", y)
	// yv := y.View(Index{0, 2, 0, 0}, Shape{2, 3, 2}) // shp: [2, 2, 2]
	// yv = yv.Permute(nil)
	// fmt.Println("yv: ", yv)
	x = x.Reshape(Shape{ComputeSize(x.shape[:x.ndims-1]), x.shape[x.ndims-1]})
	fmt.Println("x: ", x)
	it := newiterator(x)
	fmt.Println("Iterator: ", it.iscontiguous, it.istr, it.stri)
	// TODO: add this enhancement to the iterator
	// valid for depth=1 iterator
	// begin enhancement
	if !it.iscontiguous {
		rs, cs := it.x.strides[it.x.ndims-2], it.x.strides[it.x.ndims-1]
		rn, cn := it.x.shape[it.x.ndims-2], it.x.shape[it.x.ndims-1]

		if rs < cs {
			it.ds[0] = rs
			it.dn[0] = rn
			rs, cs = cs, rs
			rn, cn = cn, rn
		} else {
			it.ds[0] = cs
			it.dn[0] = cn
		}

		it.x.strides[it.x.ndims-2], it.x.strides[it.x.ndims-1] = rs, cs
		it.x.shape[it.x.ndims-2], it.x.shape[it.x.ndims-1] = rn, cn
		it.size = ComputeSize(it.x.shape[:it.x.ndims-1])
		it.istr = ComputeStrides(it.x.shape[:it.x.ndims-1])
		for i, n := range it.istr {
			it.stri[i] = 1 / float64(n)
		}
	}
	// end enhancement

	fmt.Println("ds: ", it.ds, "dn: ", it.dn)

	for xv, ok := it.init(); ok; xv, ok = it.next() {
		for j := 0; j < it.dn[0]; j++ {
			fmt.Println(xv[j*it.ds[0]])
		}
	}

	/* 	yvp := yv.Permute([]int{1, 0, 2}) // shp: [2,2,2]
	   	fmt.Println("yvp: ", yvp)
	   	yvpt := yvp.Reshape(Shape{4, 2})
	   	fmt.Println("yvp': ", yvpt)
	*/
	// it := newiterator(x, 2)
	// nxt := it.iter()
	// for v, ok := nxt(); ok; v, ok = nxt() {
	// 	for i := 0; i < it.dn[0]; i++ {
	// 		k := i * it.ds[0]
	// 		for j := 0; j < it.dn[1]; j++ {
	// 			fmt.Println(v[k+j*it.ds[1]])
	// 		}
	// 	}
	// }
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

	b.Run("closure", func(b *testing.B) {
		b.ReportAllocs()
		x := Rand(TestArrayShape)
		s := 0.0
		it := newiterator(x)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			s = 0
			nxt := it.iter()
			for xv, ok := nxt(); ok; xv, ok = nxt() {
				for _, v := range xv[:it.rn] {
					s += v * v
				}
				// for j, n := 0, it.rn; n != 0; j += it.rs {
				// 	v := xv[j]
				// 	s += v * v
				// 	n--
				// }
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
